
def test_full_records_are_arithmetically_sound(financial_records):
    for r in financial_records:
        assert r.gross_profit == r.revenue - r.cost_of_revenue
        assert r.operating_income == r.gross_profit - r.operating_expenses
        assert r.net_income == r.pre_tax_income - r.taxes
        assert r.total_assets == r.total_liabilities + r.shareholder_equity
        flows = r.cash_flow_operating + r.cash_flow_investing + r.cash_flow_financing
        assert r.ending_cash == r.beginning_cash + flows
        assert r.has_statement_detail

def test_kaggle_records_lack_statement_detail(kaggle_only_records):
    for r in kaggle_only_records:
        assert not r.has_statement_detail
        assert r.total_assets is None
        assert r.revenue is not None
