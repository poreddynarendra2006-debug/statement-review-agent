"""Generate dummy balance sheet and P&L data.

The Kaggle dataset the brief points at carries no balance sheet detail - no
total assets, liabilities, cost of revenue, taxes or cash positions - so none
of our five accounting checks can run against it. The brief anticipates this:
"Dummy balance sheet and P&L data can also be created."

This script creates that data. Three outputs:

    clean       every accounting identity holds exactly
    defective   the same statements with known errors planted in them
    labels      exactly what was planted, and where

The labels file is the important one. Without a record of what was planted,
recall cannot be computed, and "Model Performance and Evaluation" is a scored
criterion we would have no answer for.

Column names follow the Kaggle file for shared fields, so one normaliser
handles both sources. Figures are in millions USD, matching Kaggle.

Usage::

    python -m scripts.generate_dummy_statements
    python -m scripts.generate_dummy_statements --companies 80 --years 10
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SECTORS = {
    "IT": (0.62, 0.22),          # (cost ratio, opex ratio) - high margin
    "RETAIL": (0.74, 0.18),
    "MANUFACTURING": (0.71, 0.16),
    "PHARMA": (0.55, 0.28),
    "ENERGY": (0.78, 0.11),
    "BANKING": (0.48, 0.31),
    "LOGISTICS": (0.76, 0.14),
}

NAME_PARTS_A = [
    "Vertex", "Northwind", "Halcyon", "Meridian", "Kestrel", "Orion", "Sable",
    "Lumen", "Ardent", "Cobalt", "Brightwater", "Fairmont", "Ironwood",
    "Summit", "Cascade", "Beacon", "Pinnacle", "Quarry", "Thornhill", "Vantage",
]
NAME_PARTS_B = [
    "Industries", "Holdings", "Group", "Systems", "Partners", "Corporation",
    "Technologies", "Enterprises", "Labs", "Works",
]

#: Which rule each defect type breaks. Ids match the validation engine.
DEFECT_RULES = {
    "balance_sheet": "VAL_BS_01",
    "gross_profit": "VAL_GP_02",
    "operating_income": "VAL_OP_03",
    "net_income": "VAL_NI_04",
    "cash_flow": "VAL_CF_05",
}


@dataclass
class Statement:
    """One company-year, complete enough for every accounting check."""

    Year: int
    Company: str
    Category: str
    Currency: str
    Revenue: float
    Cost_of_Revenue: float
    Gross_Profit: float
    Operating_Expenses: float
    Operating_Income: float
    Pre_tax_Income: float
    Taxes: float
    Net_Income: float
    Total_Assets: float
    Current_Assets: float
    Total_Liabilities: float
    Current_Liabilities: float
    Share_Holder_Equity: float
    Cash: float
    Debt: float
    Beginning_Cash: float
    Ending_Cash: float
    Cash_Flow_from_Operating: float
    Cash_Flow_from_Investing: float
    Cash_Flow_from_Financial_Activities: float
    Earning_Per_Share: float
    EBITDA: float

    def to_row(self) -> Dict[str, object]:
        """Underscores become spaces, so headers read like the Kaggle file."""
        return {k.replace("_", " ").replace("Pre tax", "Pre-tax"): v
                for k, v in asdict(self).items()}


@dataclass
class Defect:
    """One planted error - the ground truth a benchmark scores against."""

    company: str
    year: int
    rule_id: str
    defect_type: str
    field_altered: str
    correct_value: float
    reported_value: float
    variance: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _company_names(n: int, rng: random.Random) -> List[str]:
    names = {f"{a} {b}" for a in NAME_PARTS_A for b in NAME_PARTS_B}
    return rng.sample(sorted(names), n)


def _build_company(name: str, sector: str, years: range,
                   rng: random.Random) -> List[Statement]:
    """One company's history, every identity exact."""
    cost_ratio, opex_ratio = SECTORS[sector]

    revenue = float(rng.randrange(2_000, 90_000, 100))
    equity = float(rng.randrange(1_500, 40_000, 100))
    liabilities = float(rng.randrange(1_000, 50_000, 100))
    cash = float(rng.randrange(200, 9_000, 50))
    shares = float(rng.randrange(80, 3_000))

    out: List[Statement] = []
    for year in years:
        cost = round(revenue * rng.uniform(cost_ratio - 0.04, cost_ratio + 0.04))
        gross_profit = revenue - cost
        opex = round(revenue * rng.uniform(opex_ratio - 0.03, opex_ratio + 0.03))
        operating_income = gross_profit - opex
        interest = round(abs(operating_income) * rng.uniform(0.02, 0.09))
        pre_tax = operating_income - interest
        taxes = round(pre_tax * 0.25) if pre_tax > 0 else 0.0
        net_income = pre_tax - taxes

        total_assets = liabilities + equity
        current_assets = round(total_assets * rng.uniform(0.25, 0.45))
        current_liabilities = round(liabilities * rng.uniform(0.30, 0.55))
        debt = round(liabilities * rng.uniform(0.45, 0.75))

        operating_cf = round(net_income * rng.uniform(0.85, 1.45))
        investing_cf = -round(revenue * rng.uniform(0.02, 0.09))
        financing_cf = -round(revenue * rng.uniform(0.01, 0.06))
        ending_cash = cash + operating_cf + investing_cf + financing_cf

        out.append(Statement(
            Year=year, Company=name, Category=sector, Currency="USD",
            Revenue=revenue, Cost_of_Revenue=float(cost),
            Gross_Profit=float(gross_profit),
            Operating_Expenses=float(opex),
            Operating_Income=float(operating_income),
            Pre_tax_Income=float(pre_tax), Taxes=float(taxes),
            Net_Income=float(net_income),
            Total_Assets=float(total_assets),
            Current_Assets=float(current_assets),
            Total_Liabilities=float(liabilities),
            Current_Liabilities=float(current_liabilities),
            Share_Holder_Equity=float(equity),
            Cash=float(ending_cash), Debt=float(debt),
            Beginning_Cash=float(cash), Ending_Cash=float(ending_cash),
            Cash_Flow_from_Operating=float(operating_cf),
            Cash_Flow_from_Investing=float(investing_cf),
            Cash_Flow_from_Financial_Activities=float(financing_cf),
            Earning_Per_Share=round(net_income / shares, 2),
            EBITDA=float(operating_income + round(revenue * rng.uniform(0.03, 0.08))),
        ))

        # Roll forward.
        revenue = round(revenue * rng.uniform(0.94, 1.22))
        equity = max(500.0, equity + net_income - round(net_income * rng.uniform(0, 0.4)))
        liabilities = round(liabilities * rng.uniform(0.93, 1.14))
        cash = ending_cash

    return out


def _plant(stmt: Statement, defect_type: str, rng: random.Random) -> Defect:
    """Break exactly one identity on one statement, in place.

    Each defect alters a single reported total. Nothing else is touched, so a
    planted defect maps to exactly one rule and the ground truth stays
    unambiguous.
    """
    scale = max(abs(stmt.Revenue) * rng.uniform(0.002, 0.05), 5)
    variance = round(scale) * rng.choice([1, -1])

    if defect_type == "balance_sheet":
        correct, field = stmt.Total_Assets, "Total Assets"
        stmt.Total_Assets = correct + variance
        reported = stmt.Total_Assets
    elif defect_type == "gross_profit":
        correct, field = stmt.Gross_Profit, "Gross Profit"
        stmt.Gross_Profit = correct + variance
        # Move operating income with it so only the gross profit rule breaks.
        stmt.Operating_Income += variance
        reported = stmt.Gross_Profit
    elif defect_type == "operating_income":
        correct, field = stmt.Operating_Income, "Operating Income"
        stmt.Operating_Income = correct + variance
        reported = stmt.Operating_Income
    elif defect_type == "net_income":
        correct, field = stmt.Net_Income, "Net Income"
        stmt.Net_Income = correct + variance
        reported = stmt.Net_Income
    elif defect_type == "cash_flow":
        correct, field = stmt.Ending_Cash, "Ending Cash"
        stmt.Ending_Cash = correct + variance
        reported = stmt.Ending_Cash
    else:
        raise KeyError(defect_type)

    return Defect(
        company=stmt.Company, year=stmt.Year,
        rule_id=DEFECT_RULES[defect_type], defect_type=defect_type,
        field_altered=field, correct_value=float(correct),
        reported_value=float(reported), variance=float(variance),
    )


def generate(companies: int, years: int, start_year: int,
             defect_rate: float, seed: int) -> Tuple[List[Statement], List[Statement], List[Defect]]:
    rng = random.Random(seed)
    names = _company_names(companies, rng)
    sectors = sorted(SECTORS)

    clean: List[Statement] = []
    for i, name in enumerate(names):
        clean.extend(_build_company(
            name, sectors[i % len(sectors)],
            range(start_year, start_year + years), rng,
        ))

    # Defective set is a separate copy, so the clean set stays pristine.
    defective = [Statement(**asdict(s)) for s in clean]
    defects: List[Defect] = []
    for stmt in defective:
        for defect_type in DEFECT_RULES:
            if rng.random() < defect_rate:
                defects.append(_plant(stmt, defect_type, rng))

    return clean, defective, defects


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy statement data.")
    parser.add_argument("--companies", type=int, default=60)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--defect-rate", type=float, default=0.06,
                        help="probability that any (statement, rule) pair is broken")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    clean, defective, defects = generate(
        args.companies, args.years, args.start_year, args.defect_rate, args.seed,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clean_path = out / "dummy_statements_clean.csv"
    defective_path = out / "dummy_statements_defective.csv"
    labels_path = out / "dummy_statements_labels.json"

    pd.DataFrame([s.to_row() for s in clean]).to_csv(clean_path, index=False)
    pd.DataFrame([s.to_row() for s in defective]).to_csv(defective_path, index=False)

    labels_path.write_text(json.dumps({
        "description": "Ground truth for dummy_statements_defective.csv. "
                       "Hand-independent: every entry records a defect this "
                       "script planted, not one the review engine found.",
        "generated_with": vars(args),
        "total_statements": len(defective),
        "total_defects": len(defects),
        "defects": [d.to_dict() for d in defects],
    }, indent=2), encoding="utf-8")

    print(f"clean       {len(clean):>5} statements  -> {clean_path}")
    print(f"defective   {len(defective):>5} statements  -> {defective_path}")
    print(f"defects     {len(defects):>5} planted     -> {labels_path}")

    counts: Dict[str, int] = {}
    for d in defects:
        counts[d.rule_id] = counts.get(d.rule_id, 0) + 1
    print("\ndefects by rule")
    for rule, count in sorted(counts.items()):
        print(f"  {rule}  {count:>4}")


if __name__ == "__main__":
    main()
