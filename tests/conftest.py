"""Shared test fixtures.

The orchestrator sits downstream of seven components owned by other people.
Rather than wait for those to be uploaded, this module defines fake versions
of the objects they will produce, so the pipeline and the API can be built
and tested today.

Each fake mirrors the shape of the real contract, not its behaviour. When a
teammate uploads their component, replace the corresponding fake here with a
real import and every test that used it keeps working - which is also how we
find out immediately if a contract has drifted.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fake contracts
#
# Owner is noted on each. Field names must match the real dataclass; if a
# teammate changes theirs, change it here too and the tests will tell you what
# broke.
# ---------------------------------------------------------------------------


@dataclass
class FakeFinancialRecord:
    """Owner: Data Ingestion. One company, one year.

    A superset of two sources.

    The first block is the Kaggle dataset of major companies 2009-2023, as
    confirmed by Ingestion on 9 Sep 2026. It drives trends, peer comparison
    and the machine learning work, and several of its ratios arrive already
    computed.

    The second block is the statement detail the five accounting checks need.
    The Kaggle file does not carry it, so it comes from the dummy balance
    sheet and P&L data the brief permits us to create. When a record has no
    statement detail these fields stay None and the affected rules return
    SKIPPED rather than failing - which is why ValidationResult has that
    status.
    """

    # --- Kaggle dataset fields -------------------------------------------
    year: int = 2023
    company: str = "Acme Corporation"
    category: str = "TECH"
    market_cap_b_usd: Optional[float] = 148.2
    revenue: Optional[float] = 780.0
    gross_profit: Optional[float] = 365.0
    net_income: Optional[float] = 128.0
    earnings_per_share: Optional[float] = 4.12
    ebitda: Optional[float] = 210.0
    shareholder_equity: Optional[float] = 620.0
    cash_flow_operating: Optional[float] = 118.0
    cash_flow_investing: Optional[float] = -45.0
    cash_flow_financing: Optional[float] = -38.0
    current_ratio: Optional[float] = 2.32
    debt_equity_ratio: Optional[float] = 0.77
    roe: Optional[float] = 0.2065
    roa: Optional[float] = 0.1164
    roi: Optional[float] = 0.1402
    net_profit_margin: Optional[float] = 0.1641
    free_cash_flow_per_share: Optional[float] = 2.35
    return_on_tangible_equity: Optional[float] = 0.2210
    number_of_employees: Optional[float] = 12400.0
    inflation_rate_us: Optional[float] = 3.4

    # --- Statement detail, for the five accounting checks -----------------
    currency: str = "$"
    cost_of_revenue: Optional[float] = 415.0
    operating_expenses: Optional[float] = 190.0
    operating_income: Optional[float] = 175.0
    pre_tax_income: Optional[float] = 170.0
    taxes: Optional[float] = 42.0
    total_assets: Optional[float] = 1100.0
    current_assets: Optional[float] = 360.0
    total_liabilities: Optional[float] = 480.0
    current_liabilities: Optional[float] = 155.0
    cash: Optional[float] = 210.0
    debt: Optional[float] = 100.0
    beginning_cash: Optional[float] = 175.0
    ending_cash: Optional[float] = 210.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_statement_detail(self) -> bool:
        """True when the five accounting checks can run against this record."""
        return all(
            getattr(self, name) is not None
            for name in (
                "cost_of_revenue", "operating_expenses", "pre_tax_income",
                "taxes", "total_assets", "total_liabilities",
                "beginning_cash", "ending_cash",
            )
        )


@dataclass
class FakeValidationResult:
    """Owner: Validation Agent. One accounting rule, one year."""

    rule_id: str = "VAL_BS_01"
    rule_name: str = "Balance Sheet Equality Check"
    year: int = 2023
    status: str = "PASS"                 # PASS | FAIL | SKIPPED
    expected: Optional[float] = 1100
    actual: Optional[float] = 1100
    difference: Optional[float] = 0
    severity: str = "NONE"               # NONE | LOW | MEDIUM | HIGH | CRITICAL
    evidence: str = "FY2023: Total Assets 1,100 = Liabilities 480 + Equity 620"
    formula: str = "Total Assets = Total Liabilities + Shareholders' Equity"
    message: str = "Balance sheet equation holds."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeYoYResult:
    """Owner: Trend Agent. Confirmed 10 Sep 2026.

    Note there is no is_significant flag - materiality lives on
    DeviationRecord instead, so anything asking "does this movement
    matter?" reads that, not this.
    """

    company: str = "Acme Corporation"
    year: int = 2023
    metric: str = "revenue"
    current_value: float = 780.0
    previous_value: Optional[float] = 670.0
    yoy_percent: Optional[float] = 16.42
    trend: str = "INCREASING"          # INCREASING | DECREASING | STABLE
    data_status: str = "COMPLETE"      # COMPLETE | PARTIAL | MISSING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeRatioResult:
    """Owner: Trend Agent. Confirmed 10 Sep 2026."""

    company: str = "Acme Corporation"
    year: int = 2023
    category: str = "LIQUIDITY"        # LIQUIDITY | LEVERAGE | PROFITABILITY | RETURNS
    ratio_name: str = "current_ratio"
    value: Optional[float] = 2.32
    status: str = "HEALTHY"            # HEALTHY | WARNING | CRITICAL
    formula: Optional[str] = "Current Assets / Current Liabilities"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeForecastResult:
    """Owner: Trend Agent. The trained forecasting model's prediction."""

    company: str = "Acme Corporation"
    forecast_year: int = 2024
    metric: str = "revenue"
    last_actual_year: int = 2023
    last_actual_value: float = 780.0
    forecast_value: float = 892.5
    forecast_method: str = "linear_regression"
    slope: Optional[float] = 93.5
    intercept: Optional[float] = -188_000.0
    status: str = "OK"                 # OK | INSUFFICIENT_DATA | FAILED
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeEvaluationResult:
    """Owner: Trend Agent. How well the forecasting model performs.

    These are the regression metrics the project reports for the ML work.
    """

    company: str = "Acme Corporation"
    metric: str = "revenue"
    model: str = "linear_regression"
    mae: Optional[float] = 24.8
    rmse: Optional[float] = 31.2
    r2: Optional[float] = 0.94
    evaluation_status: str = "OK"      # OK | INSUFFICIENT_DATA
    train_years_count: int = 6
    test_years_count: int = 2
    train_years_range: str = "2016-2021"
    test_years_range: str = "2022-2023"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeDeviationRecord:
    """Owner: Trend Agent. Actual against forecast - where materiality lives."""

    company: str = "Acme Corporation"
    year: int = 2023
    metric: str = "receivables"
    actual: float = 35_200.0
    forecast: float = 8_400.0
    deviation: float = 26_800.0
    deviation_percent: Optional[float] = 319.0
    absolute_deviation: float = 26_800.0
    absolute_deviation_percent: Optional[float] = 319.0
    actual_to_forecast_ratio: Optional[float] = 4.19
    direction: str = "ABOVE"           # ABOVE | BELOW | ON_TRACK
    material_deviation: bool = True
    materiality_threshold: float = 0.05
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeAnomalyFinding:
    """Owner: Anomaly Agent. Confirmed from the delivered package, 10 Sep 2026.

    Note this carries `explanation` and `recommendation` rather than the
    `evidence` field the other components use. The orchestrator normalises
    that when building the evidence packet.
    """

    company: Optional[str] = "Acme Corporation"
    year: Optional[int] = 2023
    anomaly_type: str = "historical_outlier"
    score: float = -0.0273
    severity: str = "HIGH"             # HIGH | MEDIUM | LOW
    confidence: float = 0.98
    record_id: Optional[str] = "REC_0142"
    relevant_features: Dict[str, float] = field(default_factory=dict)
    deviations: Dict[str, float] = field(default_factory=dict)
    explanation: str = (
        "The company-year observation exhibits a statistically unusual pattern."
    )
    recommendation: str = "Audit year-on-year line-item variances."
    model_name: str = "IsolationForest"
    model_mode: str = "GENERIC_LOCAL"
    percentile: Optional[float] = 97.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeRecurringIssue:
    """Owner: Anomaly Agent. A problem appearing across several years."""

    issue: str = "Operating margin declining"
    years: List[int] = field(default_factory=lambda: [2021, 2022, 2023])
    severity: str = "MEDIUM"
    evidence: str = "Operating margin fell in each of the last three years."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeRiskContributor:
    """Owner: Risk & Reporting. One component of the risk score."""

    reason: str = "Balance sheet inconsistency"
    points: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FakeRiskScoreResult:
    """Owner: Risk & Reporting. The overall 0-100 score."""

    score: int = 42
    risk_level: str = "MEDIUM"
    badge_color: str = "#B45309"
    summary: str = "Moderate risk: one material inconsistency requires review."
    contributors: List[FakeRiskContributor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "risk_level": self.risk_level,
            "badge_color": self.badge_color,
            "summary": self.summary,
            "contributors": [c.to_dict() for c in self.contributors],
        }


@dataclass
class FakeFindingItem:
    """Owner: Evidence Agent. One entry in the evidence packet."""

    finding_id: str = "FND_001"
    source: str = "validation"           # validation | trend | anomaly
    year: int = 2023
    title: str = "Balance sheet does not balance"
    severity: str = "HIGH"
    evidence: str = "Assets 1,000 against Liabilities + Equity 950. Variance 50."
    formula: str = "Total Assets = Total Liabilities + Shareholders' Equity"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def excel_bytes():
    """Build an .xlsx in memory: rows for the first sheet, plus optional named sheets."""
    import io

    from openpyxl import Workbook

    def build(*rows, sheets=None):
        book = Workbook()
        for row in rows:
            book.active.append(list(row))
        for title, sheet_rows in (sheets or {}).items():
            sheet = book.create_sheet(title)
            for row in sheet_rows:
                sheet.append(list(row))
        buffer = io.BytesIO()
        book.save(buffer)
        return buffer.getvalue()

    return build


@pytest.fixture
def financial_records() -> List[FakeFinancialRecord]:
    """Four years for one company, with full statement detail.

    Every accounting identity holds exactly, so a rule that fires against this
    fixture has found a real problem in the code rather than in the data.
    """
    rows = [
        # year  rev   cor   gp   opex  oi   pti  tax  ni   assets liab equity  cash open
        (2020, 500, 280, 220, 120, 100,  95, 25,  70,  750, 350, 400, 120,  90),
        (2021, 580, 320, 260, 140, 120, 115, 30,  85,  840, 390, 450, 145, 120),
        (2022, 670, 360, 310, 165, 145, 140, 35, 105,  960, 430, 530, 175, 145),
        (2023, 780, 415, 365, 190, 175, 170, 42, 128, 1100, 480, 620, 210, 175),
    ]
    records = []
    for (year, rev, cor, gp, opex, oi, pti, tax, ni,
         assets, liab, equity, cash, opening) in rows:
        operating = ni + 20
        investing = -35
        financing = -(opening + operating + investing - cash)
        records.append(
            FakeFinancialRecord(
                year=year, revenue=rev, cost_of_revenue=cor, gross_profit=gp,
                operating_expenses=opex, operating_income=oi,
                pre_tax_income=pti, taxes=tax, net_income=ni,
                total_assets=assets, total_liabilities=liab,
                shareholder_equity=equity, cash=cash,
                beginning_cash=opening, ending_cash=cash,
                cash_flow_operating=operating,
                cash_flow_investing=investing,
                cash_flow_financing=financing,
                net_profit_margin=round(ni / rev, 4),
                roe=round(ni / equity, 4),
                roa=round(ni / assets, 4),
            )
        )
    return records


@pytest.fixture
def kaggle_only_records() -> List[FakeFinancialRecord]:
    """Records with no statement detail, as the Kaggle file supplies them.

    The five accounting checks must return SKIPPED against these, not FAIL.
    A rule that reports a failure because a field is missing would fill the
    review with noise on every real dataset row.
    """
    blanks = dict(
        cost_of_revenue=None, operating_expenses=None, operating_income=None,
        pre_tax_income=None, taxes=None, total_assets=None,
        current_assets=None, total_liabilities=None, current_liabilities=None,
        cash=None, debt=None, beginning_cash=None, ending_cash=None,
    )
    return [
        FakeFinancialRecord(year=year, revenue=rev, gross_profit=gp,
                            net_income=ni, shareholder_equity=eq,
                            net_profit_margin=round(ni / rev, 4),
                            roe=round(ni / eq, 4), **blanks)
        for year, rev, gp, ni, eq in [
            (2020, 500, 220, 70, 400),
            (2021, 580, 260, 85, 450),
            (2022, 670, 310, 105, 530),
            (2023, 780, 365, 128, 620),
        ]
    ]


@pytest.fixture
def validation_results() -> List[FakeValidationResult]:
    """One passing check and one failing check, so both paths are exercised."""
    return [
        FakeValidationResult(),
        FakeValidationResult(
            rule_id="VAL_NI_04",
            rule_name="Net Income Check",
            status="FAIL",
            expected=128,
            actual=140,
            difference=12,
            severity="HIGH",
            evidence="FY2023: Pre-tax 170 less taxes 42 gives 128, reported 140.",
            formula="Net Income = Pre-tax Income - Taxes",
            message="Net income overstated by 12 in FY2023.",
        ),
    ]


@pytest.fixture
def yoy_results() -> List[FakeYoYResult]:
    """One ordinary movement and one large one."""
    return [
        FakeYoYResult(),
        FakeYoYResult(
            metric="receivables", previous_value=8_000.0, current_value=35_200.0,
            yoy_percent=340.0, trend="INCREASING",
        ),
    ]


@pytest.fixture
def ratio_results() -> List[FakeRatioResult]:
    """One healthy ratio and one flagged, so both paths are exercised."""
    return [
        FakeRatioResult(),
        FakeRatioResult(
            category="LEVERAGE", ratio_name="debt_to_equity", value=1.85,
            status="WARNING",
            formula="Total Liabilities / Shareholders' Equity",
        ),
    ]


@pytest.fixture
def forecasts() -> List[FakeForecastResult]:
    return [
        FakeForecastResult(),
        FakeForecastResult(metric="net_income", last_actual_value=128.0,
                           forecast_value=147.0, slope=19.4),
    ]


@pytest.fixture
def evaluations() -> List[FakeEvaluationResult]:
    """One good fit and one the model could not evaluate."""
    return [
        FakeEvaluationResult(),
        FakeEvaluationResult(metric="ebitda", mae=None, rmse=None, r2=None,
                             evaluation_status="INSUFFICIENT_DATA",
                             train_years_count=2, test_years_count=0,
                             notes="fewer than 4 periods available"),
    ]


@pytest.fixture
def deviations() -> List[FakeDeviationRecord]:
    """One material deviation and one on track."""
    return [
        FakeDeviationRecord(),
        FakeDeviationRecord(metric="revenue", actual=780.0, forecast=772.0,
                            deviation=8.0, deviation_percent=1.04,
                            absolute_deviation=8.0,
                            absolute_deviation_percent=1.04,
                            actual_to_forecast_ratio=1.01,
                            direction="ON_TRACK", material_deviation=False),
    ]


@pytest.fixture
def anomalies() -> List[FakeAnomalyFinding]:
    return [FakeAnomalyFinding()]


@pytest.fixture
def recurring_issues() -> List[FakeRecurringIssue]:
    return [FakeRecurringIssue()]


@pytest.fixture
def risk_result() -> FakeRiskScoreResult:
    return FakeRiskScoreResult(
        contributors=[
            FakeRiskContributor("Balance sheet inconsistency", 30),
            FakeRiskContributor("Receivables growth anomaly", 12),
        ]
    )


@pytest.fixture
def findings() -> List[FakeFindingItem]:
    return [FakeFindingItem()]


@pytest.fixture
def ai_summary() -> str:
    """What the Review Agent returns. Never contains an uncomputed figure."""
    return (
        "FY2023 shows one material inconsistency. Net income is reported as 140 "
        "against a computed 128, a variance of 12. Receivables rose 340% against "
        "revenue growth of 16.4%, which warrants explanation. Human review is "
        "recommended before the statements are relied upon."
    )


@pytest.fixture
def statements_df() -> pd.DataFrame:
    """A normalised dataframe, as Data Ingestion will hand it over."""
    return pd.DataFrame(
        [
            {"company": "Acme Corporation", "year": 2022, "currency": "$",
             "revenue": 670, "cost_of_revenue": 360, "gross_profit": 310,
             "operating_expenses": 165, "operating_income": 145,
             "pre_tax_income": 140, "taxes": 35, "net_income": 105,
             "total_assets": 960, "total_liabilities": 430,
             "shareholders_equity": 530, "cash": 175},
            {"company": "Acme Corporation", "year": 2023, "currency": "$",
             "revenue": 780, "cost_of_revenue": 415, "gross_profit": 365,
             "operating_expenses": 190, "operating_income": 175,
             "pre_tax_income": 170, "taxes": 42, "net_income": 128,
             "total_assets": 1100, "total_liabilities": 480,
             "shareholders_equity": 620, "cash": 210},
        ]
    )
