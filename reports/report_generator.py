"""PDF report generation for AnalysisResult.to_dict() payloads."""
from __future__ import annotations

from datetime import date
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

DISCLAIMER = "Reports potential inconsistencies for human review. Not an audit opinion and not investment advice."


def _v(result: dict[str, Any], key: str, default: Any = "") -> Any:
    value = result.get(key, default)
    return default if value is None else value


def _text(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (dict, list)):
        import json
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _para(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_text(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)


class _ReportDoc(BaseDocTemplate):
    def __init__(self, buffer: BytesIO, styles):
        super().__init__(buffer, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
                         topMargin=18 * mm, bottomMargin=20 * mm)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=self._footer)])
        self.styles_ref = styles

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawCentredString(A4[0] / 2, 8 * mm, f"Page {doc.page}  |  {DISCLAIMER}")
        canvas.restoreState()


def _table(data: list[list[Any]], widths=None, repeat_rows=1) -> Table:
    table = Table(data, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _empty(styles):
    return Paragraph("None identified", styles["BodyText"])


def generate_report(result: dict, reviewer_name: str = "") -> bytes:
    """Render a complete report from an AnalysisResult.to_dict() payload.

    The report deliberately displays supplied values only; it does not derive
    new metrics or scores from the input.
    """
    result = result or {}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=7, leading=8.5))

    buf = BytesIO()
    doc = _ReportDoc(buf, styles)
    story = []

    company = _v(result, "company", "Unknown company")
    period = _v(result, "period", "")
    risk = result.get("risk_result") or {}
    score = risk.get("score", result.get("risk_score"))
    level = risk.get("risk_level", "")

    story += [Spacer(1, 25 * mm), Paragraph("Financial Statement Review", styles["CoverTitle"]),
              Paragraph(_text(company), styles["Heading1"]), Spacer(1, 5 * mm)]
    cover = [
        [Paragraph("Period", styles["Cell"]), Paragraph(_text(period), styles["Cell"])],
        [Paragraph("Review date", styles["Cell"]), Paragraph(date.today().isoformat(), styles["Cell"])],
        [Paragraph("Risk score", styles["Cell"]), Paragraph(_text(score) if score is not None else "None", styles["Cell"])],
        [Paragraph("Risk level", styles["Cell"]), Paragraph(_text(level) or "None", styles["Cell"])],
    ]
    story += [_table(cover, [35 * mm, 125 * mm], repeat_rows=0), PageBreak()]

    story += [Paragraph("1. Executive summary", styles["Section"])]
    summary = _v(result, "ai_summary", "")
    story.append(_para(summary if summary else "No AI summary was generated.", styles["BodyText"]))
    if _v(result, "review_mode") == "heuristic":
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("This summary came from the offline reviewer.", styles["Small"]))

    story += [Paragraph("2. Risk score breakdown", styles["Section"])]
    contributors = risk.get("contributors", []) if isinstance(risk, dict) else []
    if contributors:
        rows = [[Paragraph(x, styles["Cell"]) for x in ["Reason", "Points", "Source agent"]]]
        for c in contributors:
            rows.append([_para(c.get("reason", ""), styles["Cell"]),
                         _para(c.get("points", ""), styles["Cell"]),
                         _para(c.get("source_agent", ""), styles["Cell"])])
        story.append(_table(rows, [105 * mm, 20 * mm, 35 * mm]))
    else:
        story.append(_empty(styles))

    story += [Paragraph("3. Failed validation checks", styles["Section"])]
    failed = _v(result, "failed_validations", []) or []
    if failed:
        headers = ["Rule", "Year", "Expected", "Actual", "Difference", "Formula", "Evidence"]
        rows = [[_para(h, styles["Cell"]) for h in headers]]
        for x in failed:
            rows.append([_para(x.get(k, ""), styles["Cell"]) for k in
                         ["rule_name", "year", "expected", "actual", "difference", "formula", "evidence"]])
        story.append(_table(rows, [31*mm, 12*mm, 20*mm, 20*mm, 20*mm, 28*mm, 35*mm]))
    else:
        story.append(_empty(styles))

    story += [Paragraph("4. Material deviations and anomalies", styles["Section"])]
    materials = _v(result, "material_deviations", []) or []
    anomalies = _v(result, "anomalies", []) or []
    if materials:
        story.append(Paragraph("Material deviations", styles["Heading3"]))
        rows = [[_para(h, styles["Cell"]) for h in ["Company", "Year", "Metric", "Actual", "Forecast", "Deviation", "Deviation %"]]]
        for x in materials:
            rows.append([_para(x.get(k, ""), styles["Cell"]) for k in
                         ["company", "year", "metric", "actual", "forecast", "deviation", "deviation_percent"]])
        story.append(_table(rows, [28*mm, 12*mm, 30*mm, 20*mm, 20*mm, 20*mm, 22*mm]))
    else:
        story.append(_empty(styles))
    story.append(Spacer(1, 3 * mm))
    if anomalies:
        story.append(Paragraph("Anomalies", styles["Heading3"]))
        rows = [[_para(h, styles["Cell"]) for h in ["Company", "Year", "Type", "Severity", "Score", "Confidence", "Explanation"]]]
        for x in anomalies:
            rows.append([_para(x.get(k, ""), styles["Cell"]) for k in
                         ["company", "year", "anomaly_type", "severity", "score", "confidence", "explanation"]])
        story.append(_table(rows, [25*mm, 12*mm, 27*mm, 18*mm, 17*mm, 20*mm, 41*mm]))
    else:
        story.append(_empty(styles))

    story += [Paragraph("5. Review coverage", styles["Section"])]
    coverage = _v(result, "coverage", {}) or {}
    selected = coverage.get("selected", []) or []
    skipped = coverage.get("skipped", {}) or {}
    if selected:
        story.append(Paragraph("Agents that ran: " + ", ".join(map(str, selected)), styles["BodyText"]))
    else:
        story.append(Paragraph("Agents that ran: None identified", styles["BodyText"]))
    if skipped:
        rows = [[_para("Skipped agent", styles["Cell"]), _para("Reason", styles["Cell"])]]
        for agent, reason in skipped.items():
            rows.append([_para(agent, styles["Cell"]), _para(reason, styles["Cell"])])
        story.append(Spacer(1, 3 * mm))
        story.append(_table(rows, [55*mm, 105*mm]))
    else:
        story.append(Paragraph("Skipped agents: None identified", styles["Small"]))

    security = _v(result, "security_flags", []) or []
    if security:
        story += [Paragraph("6. Security notices", styles["Section"])]
        rows = [[_para("Notice", styles["Cell"]), _para("Details", styles["Cell"])]]
        for flag in security:
            if isinstance(flag, dict):
                rows.append([_para(flag.get("flag", flag.get("type", "Security notice")), styles["Cell"]),
                             _para(flag.get("message", flag.get("description", flag)), styles["Cell"])])
            else:
                rows.append([_para("Security notice", styles["Cell"]), _para(flag, styles["Cell"])])
        story.append(_table(rows, [55*mm, 105*mm]))

    story += [Paragraph("Sign-off", styles["Section"]),
              Paragraph(f"Reviewer: {_text(reviewer_name)}", styles["BodyText"]),
              Paragraph(f"Date: {date.today().isoformat()}", styles["BodyText"]),
              Spacer(1, 8 * mm), Paragraph("Signature: ______________________________________________", styles["BodyText"])]

    doc.build(story)
    return buf.getvalue()
