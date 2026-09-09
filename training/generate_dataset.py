"""Build the training set for the review-comment model.

We do not use a hosted language model. Instead we train our own small model to
write review comments, and this script produces the data it learns from.

The supervision comes from our deterministic engine. Every example is a pair:

    input   a structured description of what the engine computed
    target  the review comment a reviewer would write about it

Because the rules are the teacher, the model never has to invent a number - it
only has to learn how to express, in professional language, something already
established as true. That is knowledge distillation from a rule-based teacher,
and it is why a 250M-parameter model is sufficient here.

Variation matters more than volume. Figures, companies, severities and
phrasings are all randomised, so the model learns the mapping rather than
memorising sentences.

Usage::

    python -m training.generate_dataset --out training/data --n 1200
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

COMPANIES = [
    "Acme Corporation", "Initech Corp", "Vertex Industries", "Northwind Traders",
    "Halcyon Manufacturing", "Meridian Foods", "Kestrel Logistics", "Orion Retail",
    "Sable Pharmaceuticals", "Lumen Energy", "Ardent Textiles", "Cobalt Mining",
]

CURRENCIES = ["INR", "USD", "EUR"]


@dataclass
class Example:
    """One training pair."""

    input: str
    target: str
    finding_type: str

    def to_dict(self) -> Dict[str, str]:
        return {"input": self.input, "target": self.target, "type": self.finding_type}


def _money(rng: random.Random, low: int, high: int) -> int:
    """A figure with realistic rounding, so the model sees plausible numbers."""
    value = rng.randint(low, high)
    return round(value, -1) if value > 1000 else value


def _fmt(value: float, currency: str) -> str:
    return f"{currency} {value:,.0f}"


# ---------------------------------------------------------------------------
# Generators - one per finding type.
#
# Each returns a structured input and several acceptable phrasings of the
# comment. Multiple targets per input teach the model that there is more than
# one correct way to say it, which stops it collapsing onto a single template.
# ---------------------------------------------------------------------------


def balance_sheet_break(rng: random.Random) -> Example:
    company = rng.choice(COMPANIES)
    currency = rng.choice(CURRENCIES)
    year = rng.randint(2015, 2024)
    liabilities = _money(rng, 200_000, 900_000)
    equity = _money(rng, 150_000, 800_000)
    variance = _money(rng, 500, 90_000) * rng.choice([1, -1])
    assets = liabilities + equity + variance
    material = abs(variance) > 0.05 * (liabilities + equity)

    inp = (
        f"finding: balance_sheet_imbalance | company: {company} | year: {year} | "
        f"rule: Total Assets = Total Liabilities + Shareholders Equity | "
        f"reported_assets: {assets} | liabilities: {liabilities} | equity: {equity} | "
        f"variance: {variance} | currency: {currency} | "
        f"materiality: {'above' if material else 'below'} threshold"
    )

    direction = "overstated" if variance > 0 else "understated"
    targets = [
        (f"The balance sheet does not balance for FY{year}. Total assets are reported as "
         f"{_fmt(assets, currency)} against liabilities and equity of "
         f"{_fmt(liabilities + equity, currency)}, a variance of {_fmt(abs(variance), currency)}. "
         f"Assets appear {direction}. Human review is recommended before the statements are relied upon."),
        (f"FY{year} shows an imbalance of {_fmt(abs(variance), currency)} between total assets "
         f"({_fmt(assets, currency)}) and the sum of liabilities and equity "
         f"({_fmt(liabilities + equity, currency)}). This breaks the fundamental accounting "
         f"equation and should be investigated before sign-off."),
        (f"Total assets for FY{year} exceed liabilities plus equity by "
         f"{_fmt(abs(variance), currency)}." if variance > 0 else
         f"Total assets for FY{year} fall short of liabilities plus equity by "
         f"{_fmt(abs(variance), currency)}.") +
        " The accounting equation must hold exactly; we recommend the underlying schedules be re-examined.",
    ]
    return Example(inp, rng.choice(targets), "balance_sheet_imbalance")


def arithmetic_break(rng: random.Random) -> Example:
    company = rng.choice(COMPANIES)
    currency = rng.choice(CURRENCIES)
    year = rng.randint(2015, 2024)

    rule, a_label, b_label, op = rng.choice([
        ("Gross Profit = Revenue - Cost of Revenue", "revenue", "cost of revenue", "-"),
        ("Operating Income = Gross Profit - Operating Expenses", "gross profit", "operating expenses", "-"),
        ("Net Income = Pre-tax Income - Taxes", "pre-tax income", "taxes", "-"),
    ])
    a = _money(rng, 100_000, 900_000)
    b = _money(rng, 20_000, min(a, 500_000))
    expected = a - b
    variance = _money(rng, 200, 40_000) * rng.choice([1, -1])
    reported = expected + variance

    inp = (
        f"finding: arithmetic_inconsistency | company: {company} | year: {year} | "
        f"rule: {rule} | {a_label}: {a} | {b_label}: {b} | "
        f"computed: {expected} | reported: {reported} | variance: {variance} | "
        f"currency: {currency}"
    )

    subject = rule.split(" =")[0]
    direction = "overstated" if variance > 0 else "understated"
    targets = [
        (f"{subject} for FY{year} does not follow from its components. With {a_label} of "
         f"{_fmt(a, currency)} and {b_label} of {_fmt(b, currency)}, the figure should be "
         f"{_fmt(expected, currency)}, but {_fmt(reported, currency)} is reported. "
         f"The line appears {direction} by {_fmt(abs(variance), currency)}."),
        (f"A variance of {_fmt(abs(variance), currency)} was identified in {subject} for FY{year}. "
         f"The reported figure of {_fmt(reported, currency)} differs from the computed "
         f"{_fmt(expected, currency)}. We recommend the supporting workings be reviewed."),
        (f"FY{year} {subject.lower()} is inconsistent with the underlying figures. "
         f"{a_label.capitalize()} less {b_label} gives {_fmt(expected, currency)}, "
         f"against {_fmt(reported, currency)} as presented."),
    ]
    return Example(inp, rng.choice(targets), "arithmetic_inconsistency")


def variance_movement(rng: random.Random) -> Example:
    company = rng.choice(COMPANIES)
    currency = rng.choice(CURRENCIES)
    year = rng.randint(2015, 2024)
    metric = rng.choice([
        "receivables", "inventory", "revenue", "operating expenses",
        "cost of revenue", "short-term debt", "cash",
    ])
    before = _money(rng, 50_000, 600_000)
    pct = rng.choice([-72, -55, -41, 38, 65, 120, 210, 340, 480])
    after = int(before * (1 + pct / 100))

    inp = (
        f"finding: material_variance | company: {company} | year: {year} | "
        f"metric: {metric} | prior_value: {before} | current_value: {after} | "
        f"percent_change: {pct} | currency: {currency} | materiality: above threshold"
    )

    move = "increased" if pct > 0 else "decreased"
    targets = [
        (f"{metric.capitalize()} {move} by {abs(pct)}% in FY{year}, from "
         f"{_fmt(before, currency)} to {_fmt(after, currency)}. A movement of this size is "
         f"material and requires explanation from management."),
        (f"A material movement was identified in {metric} for FY{year}: {abs(pct)}% "
         f"{'growth' if pct > 0 else 'reduction'} year on year "
         f"({_fmt(before, currency)} to {_fmt(after, currency)}). "
         f"We recommend obtaining supporting explanation for the change."),
        (f"FY{year} {metric} of {_fmt(after, currency)} represents a {abs(pct)}% "
         f"{'rise' if pct > 0 else 'fall'} against the prior year. "
         f"The variance exceeds our materiality threshold and warrants review."),
    ]
    return Example(inp, rng.choice(targets), "material_variance")


def cross_statement_break(rng: random.Random) -> Example:
    company = rng.choice(COMPANIES)
    currency = rng.choice(CURRENCIES)
    year = rng.randint(2015, 2024)
    bs_cash = _money(rng, 40_000, 400_000)
    variance = _money(rng, 500, 30_000) * rng.choice([1, -1])
    cf_cash = bs_cash - variance

    inp = (
        f"finding: cross_statement_mismatch | company: {company} | year: {year} | "
        f"rule: Balance sheet cash = Cash flow statement closing cash | "
        f"balance_sheet_cash: {bs_cash} | cash_flow_closing: {cf_cash} | "
        f"variance: {variance} | currency: {currency}"
    )

    targets = [
        (f"Cash does not tie between statements for FY{year}. The balance sheet reports "
         f"{_fmt(bs_cash, currency)} while the cash flow statement closes at "
         f"{_fmt(cf_cash, currency)}, a difference of {_fmt(abs(variance), currency)}. "
         f"These figures must agree."),
        (f"A cross-statement inconsistency of {_fmt(abs(variance), currency)} was identified in "
         f"FY{year} cash. Balance sheet cash of {_fmt(bs_cash, currency)} does not reconcile to "
         f"closing cash of {_fmt(cf_cash, currency)} per the cash flow statement."),
        (f"FY{year} closing cash is presented inconsistently across statements "
         f"({_fmt(bs_cash, currency)} against {_fmt(cf_cash, currency)}). "
         f"We recommend the cash reconciliation be revisited."),
    ]
    return Example(inp, rng.choice(targets), "cross_statement_mismatch")


def recurring_issue(rng: random.Random) -> Example:
    company = rng.choice(COMPANIES)
    year = rng.randint(2018, 2024)
    span = rng.randint(3, 5)
    years = list(range(year - span + 1, year + 1))
    issue = rng.choice([
        "declining operating margin", "rising leverage",
        "deteriorating current ratio", "negative operating cash flow",
        "receivables growing faster than revenue",
    ])

    inp = (
        f"finding: recurring_issue | company: {company} | "
        f"issue: {issue} | years: {','.join(str(y) for y in years)} | "
        f"consecutive_periods: {span}"
    )

    targets = [
        (f"{issue.capitalize()} has persisted across {span} consecutive reporting periods "
         f"(FY{years[0]} to FY{years[-1]}). A sustained trend of this kind is more significant "
         f"than any single year's movement and should be addressed in management commentary."),
        (f"We note {issue} in each of FY{years[0]} through FY{years[-1]}. "
         f"The persistence of this pattern over {span} periods suggests a structural rather than "
         f"a one-off issue."),
        (f"A recurring pattern was identified: {issue} across {span} periods "
         f"(FY{years[0]}-FY{years[-1]}). We recommend this be considered in the round rather "
         f"than period by period."),
    ]
    return Example(inp, rng.choice(targets), "recurring_issue")


def clean_statement(rng: random.Random) -> Example:
    """The model must also learn to say when nothing is wrong.

    Without these, it learns that every input implies a problem and will
    manufacture one.
    """
    company = rng.choice(COMPANIES)
    year = rng.randint(2015, 2024)
    checks = rng.randint(4, 6)

    inp = (
        f"finding: no_exceptions | company: {company} | year: {year} | "
        f"checks_performed: {checks} | checks_failed: 0 | "
        f"material_variances: 0"
    )

    targets = [
        (f"All {checks} arithmetic checks passed for FY{year} and no material variances were "
         f"identified. No exceptions arise from our review of the figures presented."),
        (f"FY{year} presents no exceptions. Every internal consistency check reconciled and "
         f"year-on-year movements fell within the materiality threshold."),
        (f"Our review of FY{year} identified no arithmetic inconsistencies across {checks} checks, "
         f"and no movements requiring explanation."),
    ]
    return Example(inp, rng.choice(targets), "no_exceptions")


#: Weighted so the model sees a realistic mix. Clean cases are deliberately
#: well represented - a model that never sees them over-reports.
GENERATORS: List[tuple[Callable[[random.Random], Example], int]] = [
    (arithmetic_break, 25),
    (variance_movement, 22),
    (balance_sheet_break, 18),
    (cross_statement_break, 13),
    (recurring_issue, 10),
    (clean_statement, 12),
]


def build(n: int, seed: int) -> List[Example]:
    rng = random.Random(seed)
    population = [gen for gen, weight in GENERATORS for _ in range(weight)]
    return [rng.choice(population)(rng) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review-comment training data.")
    parser.add_argument("--out", default="training/data", help="output directory")
    parser.add_argument("--n", type=int, default=1200, help="number of examples")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    examples = build(args.n, args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - args.val_split))
    train, val = examples[:split], examples[split:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, rows in (("train", train), ("validation", val)):
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"{name:<11} {len(rows):>5} examples -> {path}")

    counts: Dict[str, int] = {}
    for ex in examples:
        counts[ex.finding_type] = counts.get(ex.finding_type, 0) + 1
    print("\nfinding type distribution")
    for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<28} {count:>5}  ({count / len(examples):.1%})")


if __name__ == "__main__":
    main()
