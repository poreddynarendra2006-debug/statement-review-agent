"""Unit tests for the 3-tier semantic column mapper."""

import unittest

from src.ingestion.mapper import SemanticColumnMapper


class TestSemanticColumnMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = SemanticColumnMapper(embedding_threshold=0.65)

    def test_tier1_deterministic_mapping(self):
        raw_cols = [
            "Year",
            "Company ",
            "Category",
            "Market Cap(in B USD)",
            "Gross Profit",
            "Share Holder Equity",
            "Cash Flow from Operating",
            "Cash Flow from Financial Activities",
            "Earning Per Share",
            "Debt/Equity Ratio",
            "Net Profit Margin",
            "Number of Employees",
            "Inflation Rate(in US)",
        ]
        rename_map, results, collisions = self.mapper.map_columns(raw_cols)

        # Assert correct canonical mappings
        self.assertEqual(rename_map["Year"], "year")
        self.assertEqual(rename_map["Company "], "company")
        self.assertEqual(rename_map["Market Cap(in B USD)"], "market_cap_b_usd")
        self.assertEqual(rename_map["Gross Profit"], "gross_profit")
        self.assertEqual(rename_map["Share Holder Equity"], "shareholder_equity")
        self.assertEqual(rename_map["Cash Flow from Operating"], "cash_flow_operating")
        self.assertEqual(rename_map["Cash Flow from Financial Activities"], "cash_flow_financing")
        self.assertEqual(rename_map["Earning Per Share"], "earnings_per_share")
        self.assertEqual(rename_map["Debt/Equity Ratio"], "debt_equity_ratio")
        self.assertEqual(rename_map["Net Profit Margin"], "net_profit_margin")
        self.assertEqual(rename_map["Number of Employees"], "number_of_employees")
        self.assertEqual(rename_map["Inflation Rate(in US)"], "inflation_rate_us")
        self.assertEqual(len(collisions), 0)

        # Check that all were mapped via deterministic method (Tier 1)
        for res in results:
            self.assertEqual(res.method, "deterministic")
            self.assertEqual(res.confidence, 1.0)

    def test_tier2_embedding_semantic_mapping(self):
        # Columns that are rephrased and not in the explicit deterministic alias dictionary
        raw_cols = [
            "Year",
            "Company",
            "Revenue",
            "Net Income",
            "Stockholders Equity Value",
            "Operating Cashflow Stream",
        ]
        rename_map, results, collisions = self.mapper.map_columns(raw_cols)

        self.assertEqual(rename_map["Stockholders Equity Value"], "shareholder_equity")
        self.assertEqual(rename_map["Operating Cashflow Stream"], "cash_flow_operating")

        methods = {r.raw_name: r.method for r in results}
        self.assertEqual(methods["Stockholders Equity Value"], "embedding")
        self.assertEqual(methods["Operating Cashflow Stream"], "embedding")

    def test_tier3_llm_fallback(self):
        # Mock LLM resolver that specifically resolves an ambiguous term
        def mock_llm_resolver(raw_col, samples, candidates):
            if raw_col == "custom_opaque_profit_metric" and "gross_profit" in candidates:
                return "gross_profit"
            return None

        custom_mapper = SemanticColumnMapper(
            embedding_threshold=0.95,  # high threshold so embedding doesn't match
            llm_resolver=mock_llm_resolver,
        )

        raw_cols = ["Year", "Company", "Revenue", "Net Income", "custom_opaque_profit_metric"]
        rename_map, results, collisions = custom_mapper.map_columns(raw_cols)

        self.assertEqual(rename_map.get("custom_opaque_profit_metric"), "gross_profit")
        llm_res = [r for r in results if r.raw_name == "custom_opaque_profit_metric"][0]
        self.assertEqual(llm_res.method, "llm")

    def test_duplicate_column_collision_detection(self):
        # Both "Revenue" and "Total Revenue" map to canonical "revenue"
        raw_cols = ["Year", "Company", "Revenue", "Total Revenue", "Net Income"]
        rename_map, results, collisions = self.mapper.map_columns(raw_cols)

        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["canonical_name"], "revenue")
        self.assertEqual(rename_map["Revenue"], "revenue")
        # Secondary is preserved under normalized name, not silently overwritten
        self.assertIn("Total Revenue", rename_map)
        self.assertNotEqual(rename_map["Total Revenue"], "revenue")

    def test_unmapped_extra_column_handling(self):
        raw_cols = ["Year", "Company", "Revenue", "Net Income", "Random Extra Information 123"]
        rename_map, results, collisions = self.mapper.map_columns(raw_cols)

        unmapped = [r for r in results if r.raw_name == "Random Extra Information 123"][0]
        self.assertEqual(unmapped.method, "unmapped")
        self.assertIsNone(unmapped.canonical_name)

    def test_ambiguity_margin_escalation_to_llm(self):
        # When two candidates have very close similarity scores within ambiguity_margin,
        # the mapper should escalate to Tier 3 LLM resolver
        resolved_by_llm = []

        def mock_resolver(raw_col, samples, candidates):
            resolved_by_llm.append(raw_col)
            return candidates[0] if candidates else None

        mapper_with_margin = SemanticColumnMapper(
            embedding_threshold=0.50,
            ambiguity_margin=0.99,  # Force ambiguity check to trigger
            llm_resolver=mock_resolver,
        )

        raw_cols = ["Year", "Company", "Operating Cashflow Stream"]
        rename_map, results, collisions = mapper_with_margin.map_columns(raw_cols)

        # "Operating Cashflow Stream" was flagged as ambiguous due to margin and resolved via LLM
        stream_res = [r for r in results if r.raw_name == "Operating Cashflow Stream"][0]
        self.assertEqual(stream_res.method, "llm")
        self.assertIn("Operating Cashflow Stream", resolved_by_llm)


if __name__ == "__main__":
    unittest.main()
