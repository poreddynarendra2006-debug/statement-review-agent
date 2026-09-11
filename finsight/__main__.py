"""CLI and entry point for FinSight AI Anomaly Detection Agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from finsight.analysis.anomaly_benchmark import run_anomaly_benchmark
from finsight.analysis.anomaly_detection import AnomalyDetector
from finsight.analysis.explanation import display_name, format_evidence_value
from finsight.analysis.feature_engineering import RAW_LEVEL_REASON
from finsight.analysis.exporter import (
    export_findings_csv,
    export_findings_html,
    export_findings_json,
)
from finsight.analysis.schema_mapper import DataValidationError
from finsight.data.loader import (
    generate_reference_dataset,
    inspect_dataset,
    load_dataset,
    load_financial_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("finsight")

DEFAULT_DATASET = "data/Financial Statements.csv"


def main() -> int:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="FinSight AI - Financial Statement Anomaly Detection Agent (CSV/Excel Support)"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        nargs="?",
        const=DEFAULT_DATASET,
        help="Inspect a financial dataset (CSV/Excel) and report schema structure & feature availability.",
    )
    parser.add_argument(
        "--train",
        type=str,
        nargs="?",
        const=DEFAULT_DATASET,
        help="Train the Anomaly Detection model on a CSV/Excel dataset and save to models/.",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        nargs="?",
        const=DEFAULT_DATASET,
        help="Analyze a financial dataset (CSV/Excel) using the trained model and output findings.",
    )
    parser.add_argument(
        "--analyze-upload",
        type=str,
        help="Analyze a user-uploaded CSV or Excel financial file with schema normalization & dataset-local anomaly detection.",
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        help="Optional specific sheet name when processing multi-sheet Excel files.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the Planted Anomaly Benchmark comparing Z-score and Isolation Forest.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/isolation_forest.joblib",
        help="Path to save or load the persisted model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save findings (JSON, CSV, or HTML determined by file extension).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Optional path to export findings as tabular CSV.",
    )
    parser.add_argument(
        "--export-html",
        type=str,
        help="Optional path to export findings as a standalone HTML audit report.",
    )

    args = parser.parse_args()

    # 1. Dataset Inspection
    if args.inspect:
        csv_path = Path(args.inspect)
        if not csv_path.exists() and str(csv_path) == DEFAULT_DATASET:
            logger.info("Dataset not found at %s. Generating 23-column reference dataset...", csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            generate_reference_dataset(csv_path)

        try:
            summary = inspect_dataset(csv_path)
            loaded = load_financial_file(csv_path, sheet_name=args.sheet_name)
        except (DataValidationError, FileNotFoundError) as err:
            logger.error("Inspection Error: %s", err)
            return 1

        print("\n" + "=" * 75)
        print(f"FINSIGHT AI - DATASET INSPECTION: {summary.dataset_name}")
        print("=" * 75)
        print(f"File Type: {loaded.file_type.upper()}" + (f" (Sheet: {loaded.sheet_name})" if loaded.sheet_name else ""))
        print(f"Total Rows: {summary.num_rows} | Total Columns: {summary.num_columns}")
        print(f"Company Identifier: '{summary.company_identifier}'")
        print(f"Fiscal Year Field: '{summary.year_field}'")
        print(f"Temporal Ordering Available: {'YES' if loaded.has_temporal_ordering else 'NO (No valid year/date field)'}")
        if summary.unique_companies:
            print(f"Unique Companies ({len(summary.unique_companies)}): {', '.join(summary.unique_companies[:10])}{'...' if len(summary.unique_companies)>10 else ''}")
        if summary.years_available:
            print(f"Years Available: {min(summary.years_available)} to {max(summary.years_available)}")
        print(f"Duplicate Observations: {summary.duplicate_rows}")

        print("\n--- SCHEMA NORMALIZATION & FEATURE AVAILABILITY ---")
        print(f"Mapped Financial Columns ({len(loaded.mapping.mapped_columns)}):")
        for raw, canon in loaded.mapping.mapped_columns.items():
            print(f"  - '{raw}' -> {canon}")
        if loaded.mapping.unmapped_columns:
            print(f"Unmapped / Metadata Columns: {', '.join(loaded.mapping.unmapped_columns)}")
        print(f"\n{loaded.mapping.feature_availability_str}")
        print(f"Calculable Standard Features: {', '.join(loaded.mapping.available_features)}")
        if loaded.missing_fields:
            print(f"Missing Optional Fields: {', '.join(loaded.missing_fields)}")
        if loaded.warnings:
            print("\nWarnings / Notes:")
            for w in loaded.warnings:
                print(f"  * {w}")
        print("=" * 75)
        return 0

    # 2. Benchmark Execution
    if args.benchmark:
        print("\nRunning Planted Anomaly Benchmark...")
        results = run_anomaly_benchmark(dataset_path=DEFAULT_DATASET)
        print("\n" + "=" * 65)
        print("FINSIGHT AI - ANOMALY DETECTION BENCHMARK RESULTS")
        print("=" * 65)
        print(f"Total Observations: {results['dataset_size']} | Injected Ground-Truth Anomalies: {results['planted_anomalies_count']}")
        print(f"Features Used ({len(results['features_used'])}): {', '.join(results['features_used'])}\n")

        print(f"{'Metric':<22} | {'Z-Score Baseline':<18} | {'Isolation Forest':<18}")
        print("-" * 65)
        for metric in ["precision", "recall", "f1", "accuracy", "true_positives", "false_positives", "false_negatives"]:
            z_val = results["zscore"][metric]
            if_val = results["isolation_forest"][metric]
            print(f"{metric:<22} | {str(z_val):<18} | {str(if_val):<18}")
        print("=" * 65)
        return 0

    # 3. Model Training
    if args.train:
        train_path = Path(args.train)
        if not train_path.exists() and str(train_path) == DEFAULT_DATASET:
            logger.info("Dataset not found at %s. Generating 23-column reference dataset...", train_path)
            train_path.parent.mkdir(parents=True, exist_ok=True)
            generate_reference_dataset(train_path)

        try:
            logger.info("Loading training records from %s...", train_path)
            loaded = load_financial_file(train_path, sheet_name=args.sheet_name)
            detector = AnomalyDetector()
            summary = detector.train(loaded.records, model_mode="STANDARD_PRETRAINED")
        except (DataValidationError, FileNotFoundError, ValueError) as err:
            logger.error("Training Error: %s", err)
            return 1

        save_path = Path(args.model_path)
        detector.save(save_path)
        print("\n" + "=" * 65)
        print("FINSIGHT AI - TRAINING COMPLETE")
        print("=" * 65)
        print(f"Model Mode: {summary['model_mode']}")
        print(f"Records Processed: {summary['num_records']}")
        print(f"Engineered ML Features ({summary['num_features']}):")
        for i, feat in enumerate(summary['features'], 1):
            print(f"  {i:2d}. {feat}")
        print(f"\nInternal Anomalies Identified in Training Set: {summary['anomalies_detected_in_train']} ({summary['anomaly_rate']*100:.1f}%)")
        print(f"Model and Metadata Saved To: {save_path}")
        print("=" * 65)
        return 0

    # 4. Model Analysis / Inference (Handles both --analyze and --analyze-upload)
    is_upload_mode = bool(args.analyze_upload)
    target_file = args.analyze_upload or args.analyze
    if target_file:
        data_path = Path(target_file)
        if not data_path.exists():
            if str(data_path) == DEFAULT_DATASET:
                generate_reference_dataset(data_path)
            else:
                logger.error("Dataset not found at: %s", data_path)
                return 1

        try:
            loaded = load_financial_file(data_path, sheet_name=args.sheet_name)
        except (DataValidationError, FileNotFoundError) as err:
            logger.error("Data Validation Error: %s", err)
            return 1

        if is_upload_mode:
            logger.info("Initializing Dataset-Local Unsupervised Anomaly Detection...")
            detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
            result = detector.analyze_dataset(loaded.records, allow_local_fit=True)
        else:
            model_path = Path(args.model_path)
            if model_path.exists():
                try:
                    detector = AnomalyDetector.load(model_path)
                    logger.info("Loaded pretrained model with %d features from %s", len(detector.training_feature_names), model_path)
                except Exception as exc:
                    logger.warning("Could not load pretrained model: %s. Using dataset-local detector.", exc)
                    detector = AnomalyDetector(model_mode="GENERIC_LOCAL")
            else:
                detector = AnomalyDetector(model_mode="STANDARD_PRETRAINED")
                if str(data_path) == DEFAULT_DATASET:
                    ref_records = load_dataset(data_path)
                    detector.train(ref_records, model_mode="STANDARD_PRETRAINED")
                    detector.save(model_path)

            result = detector.analyze_dataset(loaded.records, allow_local_fit=True)

        findings = result.anomalies

        # Multi-format exports
        if args.output:
            out_file = Path(args.output)
            suffix = out_file.suffix.lower()
            if suffix == ".csv":
                export_findings_csv(findings, out_file)
            elif suffix in (".html", ".htm"):
                export_findings_html(result, out_file, title=f"FinSight AI - {loaded.source_name} Audit Report")
            else:
                export_findings_json(result, out_file)

        if args.export_csv:
            export_findings_csv(findings, Path(args.export_csv))

        if args.export_html:
            export_findings_html(result, Path(args.export_html), title=f"FinSight AI - {loaded.source_name} Audit Report")

        d_sum = result.dataset_summary
        a_sum = result.anomaly_summary

        print("\n" + "=" * 75)
        print("FINSIGHT AI - FINANCIAL ANOMALY REVIEW FINDINGS")
        print("=" * 75)
        print(f"Source File: {loaded.source_name} [{loaded.file_type.upper()}]")
        print(f"Records Evaluated: {d_sum.records}")
        print(f"Strategy Selected: {d_sum.strategy_selected}")
        print(f"Temporal Analysis: {'YES' if d_sum.temporal_analysis else 'NO (Cross-Sectional Only)'}")
        print(f"Peer / Cohort Analysis: {'YES' if d_sum.peer_analysis else 'NO (Dataset-Wide Only)'}")
        print(f"Features Available ({d_sum.features_available}): {', '.join(d_sum.available_feature_names[:10])}{'...' if len(d_sum.available_feature_names)>10 else ''}")
        if d_sum.derived_feature_names:
            print(f"Features Derived ({len(d_sum.derived_feature_names)}): {', '.join(d_sum.derived_feature_names[:8])}{'...' if len(d_sum.derived_feature_names)>8 else ''}")
        print(f"Features Actually Used ({d_sum.features_used}): {', '.join(d_sum.used_feature_names)}")
        if d_sum.ignored_features:
            print(f"Features Ignored ({len(d_sum.ignored_features)}):")
            for col, reason in list(d_sum.ignored_features.items())[:6]:
                print(f"  - {col}: {reason}")
            if len(d_sum.ignored_features) > 6:
                print(f"  - ... and {len(d_sum.ignored_features)-6} other columns.")
        size_levels = [col for col, reason in d_sum.ignored_features.items() if reason == RAW_LEVEL_REASON]
        if size_levels:
            print(
                f"Scale Neutrality: {len(size_levels)} raw company-size amounts left out of the cross-company "
                f"comparison ({', '.join(size_levels)}); ratios, margins and growth are compared instead."
            )

        print(f"\nTotal Anomalies Identified: {a_sum.total_anomalies} ({a_sum.anomaly_rate*100:.2f}% anomaly rate)")
        print(f"Severity Distribution: HIGH: {a_sum.high} | MEDIUM: {a_sum.medium} | LOW: {a_sum.low}")
        print("=" * 75)

        if loaded.warnings:
            print("\nData Notices:")
            for w in loaded.warnings:
                print(f"  * {w}")

        if not findings:
            if d_sum.message:
                print(f"\nNotice: {d_sum.message}")
            else:
                print("\nNo statistically unusual financial anomalies identified above review threshold.")
        else:
            print("\nTop Identified Anomalies For Audit Review:")
            display_limit = min(20, len(findings))
            for i, f in enumerate(findings[:display_limit], 1):
                hdr_parts = []
                if f.company:
                    hdr_parts.append(f.company)
                if f.year and f.year > 0:
                    hdr_parts.append(f"Fiscal Year {f.year}")
                if f.record_id:
                    hdr_parts.append(f"[{f.record_id}]")
                hdr_title = " - ".join(hdr_parts) if hdr_parts else (f.record_id or f"Observation #{i}")

                print(f"\n[{i}] {hdr_title}")
                print(f"    Severity: {f.severity.value} | Confidence: {f.confidence:.2f} | Anomaly Score: {f.score:.4f} | Type: {f.anomaly_type.value}")
                print(f"    Explanation: {f.explanation.replace(chr(10), ' ')}")
                print(f"    Recommendation: {f.recommendation}")
                print("    Supporting Evidence (value | robust z-score within this dataset):")
                for k, v in f.relevant_features.items():
                    dev_val = f.deviations.get(k, 0.0)
                    print(f"      - {display_name(k)}: {format_evidence_value(k, v)} | z {dev_val:+.1f}")

            if len(findings) > display_limit:
                print(f"\n... and {len(findings) - display_limit} additional anomalies for human review.")

        print("\n" + "=" * 75)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
