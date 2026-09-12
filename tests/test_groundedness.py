"""The check that every number in a written review came from the evidence.

The case that motivated this is real: our fine-tuned model wrote a cost of
revenue of 75,670 when the input said 74,670. These tests make sure that kind
of digit would be caught, and equally that legitimate phrasing is not.
"""

import pytest

from agents import groundedness
from agents.orchestrator import AnalysisResult, ReviewOrchestrator
from api.dependencies import Record


def result_with(**fields):
    return AnalysisResult(**fields)


# --- reading numbers out of text -------------------------------------------


def test_a_thousands_separator_is_the_same_number():
    assert groundedness.numbers_in("revenue of USD 105,150") == {"105150"}


def test_a_list_of_years_is_several_numbers_not_one():
    # The mistake this replaced: "2022,2023,2024" read as 202220232024, so a
    # review correctly saying FY2022 was reported as inventing it.
    assert groundedness.numbers_in("years: 2022,2023,2024") == {"2022", "2023", "2024"}


def test_a_year_written_as_a_financial_year_still_matches():
    assert "2022" in groundedness.numbers_in("across FY2022 to FY2024")


def test_sentence_punctuation_is_not_part_of_the_figure():
    assert groundedness.numbers_in("cost of USD 75,670, which is high.") == {"75670"}


def test_trailing_zeros_after_a_decimal_point_are_formatting():
    assert groundedness.numbers_in("a ratio of 2.50") == groundedness.numbers_in("a ratio of 2.5")


def test_text_with_no_numbers_reads_as_none():
    assert groundedness.numbers_in("No exceptions arise from our review.") == set()


# --- the check itself --------------------------------------------------------


def test_a_narrative_quoting_the_evidence_is_grounded():
    result = result_with(validation_results=[{"expected": 30480, "actual": 56210}])

    report = groundedness.check(
        "Gross profit should be 30,480 against 56,210 reported.", result)

    assert report.is_grounded
    assert (report.supported, report.numbers_written) == (2, 2)
    assert report.share_supported == 1.0


def test_the_digit_our_model_invented_is_caught():
    # The real failure: the evidence said 74,670 and the model wrote 75,670.
    result = result_with(validation_results=[
        {"revenue": 105150, "cost_of_revenue": 74670, "expected": 30480}])

    report = groundedness.check(
        "With revenue of USD 105,150 and cost of revenue of USD 75,670, "
        "the figure should be USD 30,480.", result)

    assert not report.is_grounded
    assert report.unsupported == ["75670"]


def test_a_difference_between_two_evidence_figures_is_allowed():
    # "overstated by 25,730" is arithmetic on figures already established,
    # which is how a review sentence is normally written.
    result = result_with(validation_results=[{"expected": 30480, "actual": 56210}])

    report = groundedness.check("The line appears overstated by 25,730.", result)

    assert report.is_grounded


def test_a_count_of_findings_is_not_an_invented_figure():
    result = result_with(
        validation_results=[{"rule_id": "VAL_BS_01"}, {"rule_id": "VAL_GP_02"}],
        companies=["Acme Corporation", "Beacon Labs", "Cobalt Mining"])

    report = groundedness.check(
        "2 checks failed across 3 companies.", result)

    assert report.is_grounded, report.unsupported


def test_an_empty_narrative_is_vacuously_grounded():
    report = groundedness.check("", result_with())

    assert (report.is_grounded, report.numbers_written, report.share_supported) == (True, 0, 1.0)


def test_a_narrative_of_pure_prose_is_grounded():
    report = groundedness.check("No exceptions arise from our review.", result_with())

    assert report.is_grounded


def test_the_share_is_reported_even_when_partly_wrong():
    result = result_with(validation_results=[{"expected": 100, "actual": 200}])

    report = groundedness.check("100 against 200, a gap of 999999.", result)

    assert report.numbers_written == 3
    assert report.supported == 2
    assert report.share_supported == pytest.approx(0.6667, abs=0.001)


def test_the_report_serialises_for_the_api():
    data = groundedness.check("100 reported.", result_with()).to_dict()

    assert set(data) == {"numbers_written", "supported", "unsupported",
                         "share_supported", "is_grounded"}


# --- through the pipeline ----------------------------------------------------


def test_a_review_quoting_a_figure_that_was_never_computed_warns_the_reviewer():
    def invents(result):
        return "Revenue of 999,999,999 was reported.", "model"

    out = ReviewOrchestrator(review=invents).run([Record(company="Acme", year=2023)])

    assert out.groundedness["is_grounded"] is False
    assert out.groundedness["unsupported"] == ["999999999"]
    assert any("not found in the computed evidence" in w for w in out.warnings)


def test_a_grounded_review_adds_no_warning():
    def writes(result):
        return f"{result.record_count} record(s) were reviewed.", "heuristic"

    out = ReviewOrchestrator(review=writes).run([Record(company="Acme", year=2023)])

    assert out.groundedness["is_grounded"] is True
    assert not [w for w in out.warnings if "not found in the computed evidence" in w]
