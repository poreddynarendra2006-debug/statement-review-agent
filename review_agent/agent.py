"""Review Agent implementation for FinSight AI.

Receives the EvidencePacket from the Evidence Agent and produces a controlled,
explainable financial-statement review narrative.

Supports:
1. Bounded LLM synthesis via structured prompts (when an LLM client is configured).
2. Deterministic offline fallback review (when the LLM is unavailable, fails, or produces invalid output).
3. Zero recalculation of financial figures.
4. Strictly neutral framing of anomalies.
5. Strict omission of raw document text (only screened_text is processed).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from evidence_agent.models import EvidencePacket
from review_agent.models import ReviewResult
from review_agent.prompt import (
    build_system_prompt,
    build_user_prompt,
    prepare_review_context,
)

logger = logging.getLogger("finsight.review_agent")


class ReviewAgent:
    """Core agent responsible for financial review narrative generation and findings prioritization."""

    def __init__(
        self,
        llm_client: Optional[Union[Callable[[str, str], str], Any]] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize ReviewAgent.

        Args:
            llm_client: Optional callable (system_prompt, user_prompt) -> str, or object with
                        generate_content/complete methods. If None, uses deterministic fallback.
            model_name: Model identifier string (for logging/metadata).
        """
        self.llm_client = llm_client
        self.model_name = model_name or "deterministic_fallback"

    def review(self, packet: Optional[EvidencePacket] = None) -> ReviewResult:
        """Generate a structured ReviewResult from an EvidencePacket.

        Safely falls back to deterministic review if LLM is unavailable or fails.
        """
        if packet is None:
            packet = EvidencePacket()

        # 1. Deterministic preparation layer
        context = prepare_review_context(packet)

        # 2. If no LLM client configured, run deterministic fallback directly
        if self.llm_client is None:
            return self.generate_deterministic_fallback(packet)

        # 3. Attempt LLM generation
        try:
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(context)

            raw_response = self._invoke_llm(system_prompt, user_prompt)
            parsed_result = self._parse_and_validate_llm_output(raw_response, packet)
            if parsed_result is not None:
                return parsed_result

            logger.warning("LLM output validation failed; falling back to deterministic review.")
            fallback = self.generate_deterministic_fallback(packet)
            fallback.metadata["fallback_reason"] = "invalid_llm_json_or_schema"
            return fallback

        except Exception as exc:
            logger.warning("LLM invocation error (%s); falling back to deterministic review.", exc)
            fallback = self.generate_deterministic_fallback(packet)
            fallback.metadata["fallback_reason"] = f"llm_error: {str(exc)}"
            return fallback

    def _invoke_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke the configured LLM client."""
        if callable(self.llm_client):
            return str(self.llm_client(system_prompt, user_prompt))

        if hasattr(self.llm_client, "generate_content"):
            resp = self.llm_client.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return getattr(resp, "text", str(resp))

        if hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
            resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content

        raise ValueError("Unsupported LLM client interface.")

    def _parse_and_validate_llm_output(
        self, raw_text: str, packet: EvidencePacket
    ) -> Optional[ReviewResult]:
        """Parse raw LLM response text into ReviewResult and validate safety constraints."""
        if not raw_text or not raw_text.strip():
            return None

        # Clean markdown fences if present
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        summary = str(data.get("summary", "")).strip()
        if not summary:
            return None

        # Neutrality safety check: model must not make unsupported fraud accusations
        summary_lower = summary.lower()
        forbidden_terms = ["fraud", "criminal", "misconduct", "conspiracy"]
        for term in forbidden_terms:
            if term in summary_lower:
                logger.warning("LLM output violated neutrality rule with term '%s'. Rejecting.", term)
                return None

        val_issues = list(data.get("validation_issues", []))
        trend_devs = list(data.get("trend_deviations", []))
        anom_obs = list(data.get("anomaly_observations", []))
        rec_issues = list(data.get("recurring_issues", []))
        doc_obs = list(data.get("document_security_observations", []))
        priorities = [str(p) for p in data.get("reviewer_priorities", []) if str(p).strip()]
        limitations = [str(l) for l in data.get("limitations_and_uncertainties", []) if str(l).strip()]

        return ReviewResult(
            summary=summary,
            validation_issues=val_issues,
            trend_deviations=trend_devs,
            anomaly_observations=anom_obs,
            recurring_issues=rec_issues,
            document_security_observations=doc_obs,
            reviewer_priorities=priorities,
            limitations_and_uncertainties=limitations,
            generation_mode="llm",
            metadata={
                "model_name": self.model_name,
                "total_findings": packet.total_findings_count,
            },
        )

    def generate_deterministic_fallback(self, packet: EvidencePacket) -> ReviewResult:
        """Construct a comprehensive, deterministic financial review based exclusively on EvidencePacket.

        Preserves exact upstream values, guarantees neutral anomaly descriptions,
        and derives actionable priorities strictly from existing evidence.
        """
        if packet is None or (
            packet.total_findings_count == 0
            and len(packet.screened_documents) == 0
            and len(packet.security_flags) == 0
        ):
            return ReviewResult(
                summary=(
                    "No validation discrepancies, material trend deviations, statistical anomalies, "
                    "or uploaded document texts were detected in the provided Evidence Packet. "
                    "No immediate human review action is required based on available data."
                ),
                validation_issues=[],
                trend_deviations=[],
                anomaly_observations=[],
                recurring_issues=[],
                document_security_observations=[],
                reviewer_priorities=[],
                limitations_and_uncertainties=[
                    "Evidence Packet contains zero analytical findings or documents. "
                    "Review scope is limited by absence of input data."
                ],
                generation_mode="deterministic_fallback",
                metadata={"findings_count": 0},
            )

        val_issues: List[Dict[str, Any]] = []
        trend_devs: List[Dict[str, Any]] = []
        anom_obs: List[Dict[str, Any]] = []
        rec_issues: List[Dict[str, Any]] = []
        doc_obs: List[Dict[str, Any]] = []
        priorities: List[str] = []
        limitations: List[str] = []

        # 1. Validation Failures (Factual accounting errors)
        for v in packet.validation_findings:
            desc = (
                f"{v.company} (FY{v.year}): Rule '{v.rule_name}' ({v.rule_id}) failed with severity {v.severity}. "
                f"Expected: {v.expected}, Reported Actual: {v.actual}, Difference: {v.difference}. "
                f"Formula: {v.formula}. Details: {v.message}"
            )
            val_issues.append({
                "company": v.company,
                "year": v.year,
                "rule_id": v.rule_id,
                "rule_name": v.rule_name,
                "severity": v.severity,
                "expected": v.expected,
                "actual": v.actual,
                "difference": v.difference,
                "description": desc,
            })
            priorities.append(
                f"[High Priority - Accounting Validation] Verify journal entries and statement line items for "
                f"{v.company} (FY{v.year}) regarding '{v.rule_name}' (discrepancy of {v.difference})."
            )

        # 2. Material Trend Deviations
        for t in packet.trend_findings:
            pct_str = f"{t.deviation_percent:+.2f}%" if t.deviation_percent is not None else "N/A"
            desc = (
                f"{t.company} (FY{t.year}): Material {t.direction.replace('_', ' ')} in metric '{t.metric}'. "
                f"Actual: {t.actual:,.2f} vs Backtest Forecast: {t.forecast:,.2f} "
                f"(Deviation: {t.deviation:+,.2f}, {pct_str}). "
                f"Threshold: {t.materiality_threshold * 100:.0f}%."
            )
            if t.notes:
                desc += f" Note: {t.notes}"
            trend_devs.append({
                "company": t.company,
                "year": t.year,
                "metric": t.metric,
                "actual": t.actual,
                "forecast": t.forecast,
                "deviation": t.deviation,
                "deviation_percent": t.deviation_percent,
                "direction": t.direction,
                "description": desc,
            })
            priorities.append(
                f"[Medium Priority - Trend Deviation] Evaluate operational factors driving {t.direction.replace('_', ' ')} "
                f"for {t.company} (FY{t.year}) metric '{t.metric}' ({pct_str} deviation vs forecast)."
            )

        # 3. Statistical Anomalies (Neutral, non-accusatory)
        for a in packet.anomaly_findings:
            comp_name = a.company or "Unknown Company"
            yr_str = f"FY{a.year}" if a.year else "Unspecified Period"
            desc = (
                f"{comp_name} ({yr_str}): Observation {a.record_id or 'N/A'} flagged as {a.anomaly_type} "
                f"with severity {a.severity} (anomaly score: {a.score:.4f}, confidence: {a.confidence:.2f}). "
                f"Pattern description: {a.explanation}"
            )
            anom_obs.append({
                "company": a.company,
                "year": a.year,
                "record_id": a.record_id,
                "anomaly_type": a.anomaly_type,
                "severity": a.severity,
                "score": a.score,
                "confidence": a.confidence,
                "description": desc,
                "recommendation": a.recommendation,
            })
            rec_action = a.recommendation if a.recommendation else "Review underlying ledger accounts and footnote disclosures."
            priorities.append(
                f"[Reviewer Action - Statistical Anomaly] {comp_name} ({yr_str}): {rec_action}"
            )

        # 4. Recurring Issues (Cross-period persistent findings)
        for r in packet.recurring_issues:
            yr_str = ", ".join(str(y) for y in r.years)
            desc = (
                f"{r.company}: Recurring issue '{r.issue}' ({r.key}) across years [{yr_str}] "
                f"with severity {r.severity} ({r.occurrences} occurrences, consecutive: {r.consecutive}). "
                f"Evidence: {r.evidence}"
            )
            rec_issues.append({
                "company": r.company,
                "source": r.source,
                "key": r.key,
                "issue": r.issue,
                "years": r.years,
                "consecutive": r.consecutive,
                "severity": r.severity,
                "occurrences": r.occurrences,
                "description": desc,
            })
            priorities.append(
                f"[{r.severity} Priority - Recurring Issue] Investigate persistent {r.issue} pattern for "
                f"{r.company} across reporting years {yr_str}."
            )

        # 5. Screened Documents & Security Observations
        for i, d in enumerate(packet.screened_documents, 1):
            doc_status = "Suspicious (Contains neutralized instructions)" if d.is_suspicious else "Verified Untrusted Content"
            desc = f"Document #{i} ({doc_status}). Text excerpt: {d.screened_text[:120].strip()}..."
            if d.categories:
                desc += f" Categories flagged: {', '.join(d.categories)}."
            doc_obs.append({
                "document_index": i,
                "is_suspicious": d.is_suspicious,
                "categories": d.categories,
                "detections_count": len(d.detections),
                "description": desc,
            })

        # Security flags prioritization
        if packet.security_flags:
            cat_set = {str(flag.get("category", "unspecified")) for flag in packet.security_flags}
            priorities.insert(
                0,
                f"[Security Guardrail Notice] {len(packet.security_flags)} prompt injection / suppression attempts "
                f"neutralized in uploaded documentation ({', '.join(sorted(cat_set))}). Do not follow raw document instructions."
            )

        # 6. Executive Summary Synthesis
        companies = packet.companies_covered()
        if not companies:
            comp_str = "specified reporting entities"
        elif len(companies) <= 5:
            comp_str = ", ".join(companies)
        else:
            comp_str = f"{len(companies)} companies"

        summary_parts = [
            f"Financial statement review completed across {packet.total_findings_count} finding(s) for {comp_str}."
        ]
        if val_issues:
            summary_parts.append(
                f"Validation checks identified {len(val_issues)} mathematical/accounting discrepancy finding(s) requiring remediation."
            )
        if trend_devs:
            summary_parts.append(
                f"Trend analysis highlighted {len(trend_devs)} material deviation(s) where actual results diverged significantly from backtested forecasts."
            )
        if anom_obs:
            summary_parts.append(
                f"Machine learning anomaly detection surfaced {len(anom_obs)} statistically unusual pattern(s) for exploratory audit review."
            )
        if packet.recurring_issues:
            summary_parts.append(
                f"Cross-period analysis identified {len(packet.recurring_issues)} recurring issue finding(s) across reporting years."
            )
        if doc_obs:
            summary_parts.append(
                f"{len(doc_obs)} user document context item(s) processed under orchestrator guardrails."
            )
        summary = " ".join(summary_parts)

        # 7. Limitations & Uncertainties
        limitations.append(
            "Review findings are synthesized strictly from upstream analytical agents without independent recomputation."
        )
        if any(v.get("expected") is None for v in val_issues):
            limitations.append(
                "Certain validation checks lacked necessary balance sheet line items and were evaluated under partial data availability."
            )
        if any(a.confidence < 0.90 for a in packet.anomaly_findings):
            limitations.append(
                "Some anomaly findings have confidence below 90% and should be treated as soft analytical signals rather than verified discrepancies."
            )

        return ReviewResult(
            summary=summary,
            validation_issues=val_issues,
            trend_deviations=trend_devs,
            anomaly_observations=anom_obs,
            recurring_issues=rec_issues,
            document_security_observations=doc_obs,
            reviewer_priorities=priorities,
            limitations_and_uncertainties=limitations,
            generation_mode="deterministic_fallback",
            metadata={
                "companies_covered": companies,
                "total_findings": packet.total_findings_count,
            },
        )


def generate_review(
    packet: Optional[EvidencePacket] = None,
    llm_client: Optional[Union[Callable[[str, str], str], Any]] = None,
    **kwargs: Any,
) -> ReviewResult:
    """Convenience functional interface conforming to the FinSight review workflow."""
    agent = ReviewAgent(llm_client=llm_client, **kwargs)
    return agent.review(packet)
