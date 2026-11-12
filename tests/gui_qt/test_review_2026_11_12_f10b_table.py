"""parse_table_text header sniffing: a mixed first row is data, not a header."""

import pytest

from nlsq.gui_qt.adapters.data_adapter import parse_table_text


def test_first_row_with_bad_cell_is_not_swallowed_as_header():
    with pytest.raises(ValueError, match="line 1"):
        parse_table_text("1.0,NA\n2.0,4.0\n3.0,6.0\n")


@pytest.mark.parametrize(
    "text,header",
    [
        ("x,y\n1,2\n3,4\n", ["x", "y"]),
        ("time (s),signal\n1,2\n3,4\n", ["time (s)", "signal"]),
        ("1st,2nd\n1,2\n3,4\n", ["1st", "2nd"]),
        ("time\tsignal\n1\t2\n3\t4\n", ["time", "signal"]),
        ("x y\n1 2\n3 4\n", ["x", "y"]),
    ],
)
def test_non_numeric_headers_still_recognised(text, header):
    rows, got = parse_table_text(text)
    assert got == header
    assert rows == [[1.0, 2.0], [3.0, 4.0]]


def test_header_with_numeric_looking_cell_is_data():
    with pytest.raises(ValueError, match="line 1"):
        parse_table_text("run1,2\n1,2\n")


def test_error_line_numbers_are_file_lines_after_header():
    with pytest.raises(ValueError, match=r"line 3 .*'3,x'"):
        parse_table_text("x,y\n1,2\n3,x\n")
    # semicolon CSV, decimal comma: header consumed line 1, bad row is line 2
    with pytest.raises(ValueError, match=r"line 2 "):
        parse_table_text("x;y\n1,5;a\n")


def test_headerless_error_line_number_unchanged():
    with pytest.raises(ValueError, match="line 2"):
        parse_table_text("1,2\n3,x\n")
