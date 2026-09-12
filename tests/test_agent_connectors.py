"""The Evidence and Review connectors, tested without their packages installed.

Both are stubbed here, so these tests state the contract this system relies on:
what shape it hands over, and what it expects back. If the Evidence role's
package changes shape, one of these fails and says which half moved.
"""

import importlib
import sys
import types

import pytest

from agents.orchestrator import AnalysisResult

# The connectors import the Evidence role's packages at module level, so that a
# missing package means "not installed" rather than a connector that fails on
# every review. The fixtures below put stand-ins in place first, then import.


class FakeItem:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def to_dict(self):
        return dict(self.__dict__)


class FakePacket:
    def __init__(self, **groups):
        self.__dict__.update(groups)


class FakeReview:
    def __init__(self, summary, generation_mode):
        self.summary = summary
        self.generation_mode = generation_mode


@pytest.fixture
def evidence_module(monkeypatch):
    """Stand in for the Evidence role's package."""
    module = types.ModuleType("evidence_agent")
    module.packet = FakePacket()
    module.seen = []

    def compile_all_findings(result):
        module.seen.append(result)
        return module.packet

    module.compile_all_findings = compile_all_findings
    monkeypatch.setitem(sys.modules, "evidence_agent", module)
    return module


@pytest.fixture
def evidence_agent(evidence_module):
    """The connector, imported against the stand-in package."""
    return importlib.reload(importlib.import_module("agents.evidence_agent"))


@pytest.fixture
def review_module(monkeypatch):
    """Stand in for the Review role's package."""
    module = types.ModuleType("review_agent")
    module.result = FakeReview("Two matters need attention.", "deterministic_fallback")
    module.generate_review = lambda packet: module.result
    monkeypatch.setitem(sys.modules, "review_agent", module)
    return module


@pytest.fixture
def review_agent(review_module, evidence_module):
    """The connector, imported against both stand-in packages."""
    return importlib.reload(importlib.import_module("agents.review_agent"))


def test_findings_come_back_flat_and_in_reading_order(evidence_module, evidence_agent):
    evidence_module.packet = FakePacket(
        validation_findings=[FakeItem(source="validation", rule_id="VAL_BS_01")],
        trend_findings=[FakeItem(source="trend", metric="revenue")],
        anomaly_findings=[FakeItem(source="anomaly", anomaly_type="margin_shift")],
    )

    findings = evidence_agent.compile_all_findings(AnalysisResult())

    assert [f["source"] for f in findings] == ["validation", "trend", "anomaly"]
    assert findings[0]["rule_id"] == "VAL_BS_01"


def test_recurring_issues_flow_through_once_the_packet_carries_them(evidence_module, evidence_agent):
    evidence_module.packet = FakePacket(
        validation_findings=[FakeItem(source="validation")],
        recurring_findings=[FakeItem(source="recurring", years=[2017, 2018, 2019])],
    )

    findings = evidence_agent.compile_all_findings(AnalysisResult())

    assert findings[-1]["years"] == [2017, 2018, 2019]


def test_a_packet_with_no_findings_gives_an_empty_list(evidence_module, evidence_agent):
    assert evidence_agent.compile_all_findings(AnalysisResult()) == []


def test_dictionaries_and_a_plain_list_are_both_accepted(evidence_module, evidence_agent):
    evidence_module.packet = [{"source": "validation"}, FakeItem(source="trend")]

    findings = evidence_agent.compile_all_findings(AnalysisResult())

    assert [f["source"] for f in findings] == ["validation", "trend"]


def test_the_whole_result_is_handed_over_not_a_copy(evidence_module, evidence_agent):
    result = AnalysisResult(company="Halcyon Group")

    evidence_agent.compile_all_findings(result)

    assert evidence_module.seen == [result]


@pytest.mark.parametrize("their_mode, ours", [
    ("llm", "model"),
    ("model", "model"),
    ("deterministic_fallback", "heuristic"),
    ("heuristic", "heuristic"),
    ("", "heuristic"),
    ("something new", "heuristic"),
])
def test_review_mode_is_reported_in_this_systems_terms(
        evidence_module, review_module, review_agent, their_mode, ours):
    review_module.result = FakeReview("A summary.", their_mode)

    summary, mode = review_agent.write_review(AnalysisResult())

    assert (summary, mode) == ("A summary.", ours)


def test_a_review_returned_as_plain_text_still_works(evidence_module, review_module, review_agent):
    review_module.generate_review = lambda packet: "Plain narrative."

    assert review_agent.write_review(AnalysisResult()) == ("Plain narrative.", "heuristic")


def test_write_review_is_preferred_if_they_ever_rename_it(evidence_module, review_module, review_agent):
    review_module.write_review = lambda packet: FakeReview("Renamed.", "llm")

    assert review_agent.write_review(AnalysisResult()) == ("Renamed.", "model")


def test_the_connector_is_absent_until_their_package_is_installed():
    # How the API knows the component is not installed yet: importing the
    # connector must fail, rather than wiring one that raises on every review.
    for name in ("evidence_agent", "review_agent", "agents.evidence_agent", "agents.review_agent"):
        sys.modules.pop(name, None)

    with pytest.raises(ImportError):
        importlib.import_module("agents.evidence_agent")


def test_a_failing_review_does_not_discard_the_findings(
        evidence_module, review_module, evidence_agent, review_agent):
    def explode(packet):
        raise RuntimeError("model unavailable")

    review_module.generate_review = explode
    evidence_module.packet = FakePacket(validation_findings=[FakeItem(source="validation")])

    from agents.orchestrator import ReviewOrchestrator
    from api.dependencies import Record

    result = ReviewOrchestrator(
        evidence=evidence_agent.compile_all_findings,
        review=review_agent.write_review,
    ).run([Record(company="Halcyon Group", year=2019)])

    assert len(result.findings) == 1
    assert result.review_mode == "none"
    assert any("model unavailable" in w for w in result.warnings)
