"""Excel uploads must reach Data Ingestion as the table they contain."""

import pytest

from api.spreadsheets import NoTableFound, excel_to_csv


def lines(content):
    return content.decode("utf-8").splitlines()


def test_a_plain_sheet_converts_as_is(excel_bytes):
    workbook = excel_bytes(("Year", "Company", "Revenue"), (2023, "Acme", 780))

    assert lines(excel_to_csv(workbook)) == ["Year,Company,Revenue", "2023,Acme,780"]


def test_titles_units_and_blank_rows_above_the_header_are_dropped(excel_bytes):
    workbook = excel_bytes(("Nova Textiles Ltd",), ("Annual figures, INR lakh",), (),
                           ("Fiscal Year", "Company Name", "Total Sales"), (2023, "Nova", 5000))

    assert lines(excel_to_csv(workbook)) == ["Fiscal Year,Company Name,Total Sales", "2023,Nova,5000"]


def test_the_sheet_holding_the_figures_wins_over_a_notes_sheet(excel_bytes):
    workbook = excel_bytes(
        ("Item", "Note"), ("Basis", "Prepared under Ind AS"),
        sheets={"Figures": [("Year", "Company", "Revenue"), (2022, "Acme", 670), (2023, "Acme", 780)]},
    )

    assert lines(excel_to_csv(workbook))[0] == "Year,Company,Revenue"


def test_a_workbook_with_no_table_raises(excel_bytes):
    with pytest.raises(NoTableFound):
        excel_to_csv(excel_bytes(("Only a title",)))
