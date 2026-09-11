"""FinSight Analysis module - Universal Anomaly Detection, Dynamic Profiling, Feature Engineering, and Evaluation."""

from finsight.analysis.anomaly_detection import (
    AnomalyDetector,
    IsolationForestModel,
    ZScoreBaseline,
    detect_anomalies,
    load_model,
    save_model,
    train_model,
)
from finsight.analysis.anomaly_features import FeaturePipeline, compute_financial_features
from finsight.analysis.anomaly_scoring import (
    calibrate_continuous_score,
    compute_confidence_scores,
    grade_severities,
)
from finsight.analysis.dataset_profiler import profile_dataset
from finsight.analysis.explanation import (
    build_anomaly_finding,
    categorize_anomaly_type,
    format_feature_value,
    generate_grounded_explanation,
    generate_recommendation,
)
from finsight.analysis.feature_engineering import extract_dynamic_features
from finsight.analysis.schema_mapper import (
    CANONICAL_FIELDS,
    DataValidationError,
    SchemaMapper,
    SchemaMappingResult,
    normalize_header,
)
from finsight.analysis.exporter import (
    export_findings_csv,
    export_findings_html,
    export_findings_json,
)
from finsight.analysis.strategy_selector import select_strategy

__all__ = [
    "FeaturePipeline",
    "compute_financial_features",
    "extract_dynamic_features",
    "profile_dataset",
    "select_strategy",
    "calibrate_continuous_score",
    "compute_confidence_scores",
    "grade_severities",
    "build_anomaly_finding",
    "categorize_anomaly_type",
    "format_feature_value",
    "generate_grounded_explanation",
    "generate_recommendation",
    "AnomalyDetector",
    "ZScoreBaseline",
    "IsolationForestModel",
    "train_model",
    "detect_anomalies",
    "save_model",
    "load_model",
    "CANONICAL_FIELDS",
    "DataValidationError",
    "SchemaMapper",
    "SchemaMappingResult",
    "normalize_header",
    "export_findings_csv",
    "export_findings_html",
    "export_findings_json",
]
