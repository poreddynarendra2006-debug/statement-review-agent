"""Multi-format Export Layer for FinSight AI Anomaly Detection Agent.

Provides robust serializers to export analysis findings into:
1. Standard JSON (complete model and finding provenance)
2. Tabular CSV (for spreadsheet and audit workpapers)
3. Self-contained HTML report (modern audit-grade interactive visualization)
"""

from __future__ import annotations

import csv
import html
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from finsight.analysis.explanation import display_name, format_evidence_value
from finsight.core.models import AnalysisResult, AnomalyFinding, Severity

logger = logging.getLogger(__name__)


def export_findings_json(result: AnalysisResult, file_path: Union[str, Path]) -> None:
    """Export the complete AnalysisResult to a formatted JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, indent=2)
    logger.info("Exported JSON analysis result to: %s", path)


def export_findings_csv(findings: Sequence[AnomalyFinding], file_path: Union[str, Path]) -> None:
    """Export anomaly findings to a standard CSV file for spreadsheet consumption."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "record_id",
        "company",
        "year",
        "severity",
        "confidence",
        "score",
        "anomaly_type",
        "contributing_features",
        "explanation",
        "recommendation",
    ]

    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)

        for f in findings:
            sev_str = f.severity.value if isinstance(f.severity, Severity) else str(f.severity)
            type_str = f.anomaly_type.value if hasattr(f.anomaly_type, "value") else str(f.anomaly_type)
            feats_str = "; ".join(
                f"{k}={v:.4f} (robust z {f.deviations.get(k, 0.0):+.2f})" for k, v in f.relevant_features.items()
            )

            writer.writerow([
                f.record_id or "",
                f.company or "",
                f.year if (f.year is not None and f.year > 0) else "",
                sev_str,
                f"{f.confidence:.2f}",
                f"{f.score:.4f}",
                type_str,
                feats_str,
                f.explanation or "",
                f.recommendation or "",
            ])

    logger.info("Exported %d anomaly findings to CSV: %s", len(findings), path)


def export_findings_html(
    result: AnalysisResult,
    file_path: Union[str, Path],
    title: str = "FinSight AI - Financial Anomaly Audit Report",
) -> None:
    """Generate a self-contained, responsive HTML audit report with visual badges and summary cards."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    d_sum = result.dataset_summary
    a_sum = result.anomaly_summary
    findings = result.anomalies

    # Severity badge styles
    sev_styles = {
        "HIGH": "background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171;",
        "MEDIUM": "background-color: #fef3c7; color: #92400e; border: 1px solid #fbbf24;",
        "LOW": "background-color: #e0f2fe; color: #075985; border: 1px solid #38bdf8;",
    }

    # Generate table rows
    rows_html: list[str] = []
    if not findings:
        rows_html.append(
            '<tr><td colspan="9" style="text-align: center; padding: 24px; color: #64748b;">'
            'No statistically unusual anomalies detected exceeding threshold.</td></tr>'
        )
    else:
        for idx, f in enumerate(findings, 1):
            sev_key = f.severity.value if isinstance(f.severity, Severity) else str(f.severity).upper()
            badge_style = sev_styles.get(sev_key, "background-color: #f1f5f9; color: #334155;")
            entity_str = html.escape(str(f.company or f"Record {f.record_id}"))
            period_str = f" ({f.year})" if (f.year and f.year > 0) else ""

            feat_pills = "".join(
                f'<span class="feat-pill" title="robust z-score: {f.deviations.get(k, 0.0):+.2f}">'
                f'{html.escape(display_name(k))}: {html.escape(format_evidence_value(k, v))}</span>'
                for k, v in list(f.relevant_features.items())[:3]
            )

            a_type = f.anomaly_type.value if hasattr(f.anomaly_type, "value") else str(f.anomaly_type)
            type_label = html.escape(a_type.replace("_", " ").title())

            rows_html.append(f"""
            <tr>
                <td style="font-weight: 600; color: #475569;">#{idx}</td>
                <td><strong>{entity_str}</strong>{period_str}</td>
                <td><span class="badge" style="{badge_style}">{sev_key}</span></td>
                <td>{f.confidence:.2f}</td>
                <td>{f.score:.4f}</td>
                <td><span class="type-tag">{type_label}</span></td>
                <td style="max-width: 220px;">{feat_pills}</td>
                <td style="max-width: 320px; font-size: 0.88rem;">{html.escape(f.explanation)}</td>
                <td style="max-width: 280px; font-size: 0.88rem; color: #334155;">{html.escape(f.recommendation)}</td>
            </tr>
            """)

    table_body = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --primary: #0f172a;
            --text: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --primary-accent: #2563eb;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 28px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 16px;
        }}
        h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 4px;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .card-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
        }}
        .card-value {{
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 4px;
            color: var(--primary);
        }}
        .meta-bar {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 28px;
            font-size: 0.88rem;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }}
        .meta-item strong {{ color: var(--primary); }}
        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow-x: auto;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}
        th {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}
        tr:hover td {{
            background-color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}
        .type-tag {{
            display: inline-block;
            background: #f1f5f9;
            color: #475569;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .feat-pill {{
            display: inline-block;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            color: #1e293b;
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 2px 2px 2px 0;
            font-family: monospace;
        }}
        footer {{
            margin-top: 32px;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .card {{ box-shadow: none; border: 1px solid #ccc; }}
            .table-container {{ box-shadow: none; border: 1px solid #ccc; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>{html.escape(title)}</h1>
        <div class="subtitle">FinSight AI Autonomous Financial Statement Anomaly Detection Audit</div>
    </header>

    <div class="cards-grid">
        <div class="card">
            <div class="card-label">Records Evaluated</div>
            <div class="card-value">{d_sum.records}</div>
        </div>
        <div class="card">
            <div class="card-label">Identified Anomalies</div>
            <div class="card-value">{a_sum.total_anomalies}</div>
        </div>
        <div class="card">
            <div class="card-label">Anomaly Rate</div>
            <div class="card-value">{a_sum.anomaly_rate*100:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-label">High Severity</div>
            <div class="card-value" style="color: #b91c1c;">{a_sum.high}</div>
        </div>
        <div class="card">
            <div class="card-label">Medium Severity</div>
            <div class="card-value" style="color: #b45309;">{a_sum.medium}</div>
        </div>
        <div class="card">
            <div class="card-label">Low Severity</div>
            <div class="card-value" style="color: #0369a1;">{a_sum.low}</div>
        </div>
    </div>

    <div class="meta-bar">
        <div class="meta-item"><strong>Strategy:</strong> {html.escape(d_sum.strategy_selected or "Adaptive")}</div>
        <div class="meta-item"><strong>Temporal Ordering:</strong> {"Yes" if d_sum.temporal_analysis else "No"}</div>
        <div class="meta-item"><strong>Features Used:</strong> {d_sum.features_used}</div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Entity / Period</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                    <th>Score</th>
                    <th>Anomaly Type</th>
                    <th>Key Contributing Features</th>
                    <th>Audit Explanation</th>
                    <th>Actionable Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {table_body}
            </tbody>
        </table>
    </div>

    <footer>
        Generated by FinSight AI Anomaly Detection Agent &bull; All evaluations strictly conservative and data-grounded.
    </footer>
</div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(html_content)
    logger.info("Exported HTML audit report to: %s", path)
