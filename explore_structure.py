import openpyxl
import pandas as pd
import os
import sys

FOLDER = r"C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
OUTPUT = os.path.join(FOLDER, "data_structure_analysis.txt")
FILE1 = os.path.join(FOLDER, "Anti roll bar-1.xlsx")

out_lines = []

def w(msg=""):
    out_lines.append(msg)
    print(msg)

w("=" * 120)
w("EXCEL STRUCTURE ANALYSIS: Anti roll bar-1.xlsx")
w("=" * 120)
w()

# --- openpyxl for full detail ---
wb = openpyxl.load_workbook(FILE1, data_only=True, read_only=False)

w(f"File: Anti roll bar-1.xlsx")
w(f"Sheet names: {wb.sheetnames}")
w(f"Workbook properties: {wb.properties.title}, {wb.properties.subject}, {wb.properties.description}")
w()

for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    w("-" * 120)
    w(f"SHEET: '{sheet_name}'")
    w(f"  Dimensions: {ws.dimensions}")
    w(f"  Min row: {ws.min_row}, Max row: {ws.max_row}")
    w(f"  Min col: {ws.min_column}, Max col: {ws.max_column}")
    w(f"  Column count (max column index): {ws.max_column}")

    # Read first 25 rows to understand headers and data
    w()
    w(f"  First 25 rows (all columns):")
    w()

    rows_data = []
    headers = None

    for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=min(25, ws.max_row), values_only=False), start=1):
        row_values = [(cell.column_letter, cell.column, cell.value, cell.data_type) for cell in row]
        rows_data.append(row_values)

        if row_idx == 1:
            headers = [cell.value for cell in row]
            w(f"    Row {row_idx} [HEADERS]:")
            for col_letter, col_num, val, dtype in row_values:
                w(f"      {col_letter}(col{col_num}): {val!r}  [type: {dtype}]")
        else:
            w(f"    Row {row_idx}:")
            for col_letter, col_num, val, dtype in row_values:
                # Truncate long values
                display_val = str(val)
                if len(display_val) > 120:
                    display_val = display_val[:120] + "..."
                w(f"      {col_letter}(col{col_num}): {display_val!r}  [type: {dtype}]")

    w()
    # Determine which rows look like headers vs data
    # Find rows where most cells have non-None values (likely header rows)
    w(f"  COLUMN SUMMARY:")
    for col_idx in range(1, ws.max_column + 1):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        col_values = []
        for row in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=1, max_row=ws.max_row, values_only=False):
            for cell in row:
                col_values.append((cell.row, cell.value, cell.data_type))

        # Show first few and last few
        non_none = [(r, v, t) for r, v, t in col_values if v is not None]
        w(f"    Column {col_letter} (col {col_idx}): {len(non_none)} non-empty values out of {ws.max_row} rows")
        if non_none:
            first_few = non_none[:3]
            last_few = non_none[-3:]
            w(f"      First values: {[(r, repr(str(v)[:60]), t) for r, v, t in first_few]}")
            if len(non_none) > 6:
                w(f"      Last values:  {[(r, repr(str(v)[:60]), t) for r, v, t in last_few]}")

    w()
    # Check for trace-like columns (long strings, lists, many data points per cell)
    w(f"  TRACE DATA DETECTION (cells with large amounts of data):")
    trace_candidates = []
    for col_idx in range(1, ws.max_column + 1):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        for cell in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=1, max_row=min(50, ws.max_row), values_only=False):
            for c in cell:
                if c.value is not None:
                    s = str(c.value)
                    if len(s) > 200:
                        trace_candidates.append((col_letter, c.row, len(s), repr(s[:100])))

    if trace_candidates:
        for col_letter, row, length, preview in trace_candidates[:20]:
            w(f"    Cell {col_letter}{row}: {length} chars, preview: {preview}")
    else:
        w("    No trace data cells found (no cell > 200 chars in first 50 rows)")

    w()

# Also read with pandas for type analysis
w("=" * 120)
w("PANDAS ANALYSIS (for comparison)")
w("=" * 120)

for sheet_name in wb.sheetnames:
    w()
    w(f"SHEET: '{sheet_name}'")
    try:
        # Try reading with header from different rows
        for header_row in [0, 1, 2]:
            try:
                df = pd.read_excel(FILE1, sheet_name=sheet_name, header=header_row, nrows=30)
                w(f"  --- header={header_row} (row {header_row+1} in Excel) ---")
                w(f"  Columns: {list(df.columns)}")
                w(f"  dtypes:\n{df.dtypes}")
                w(f"  Shape: {df.shape}")
                w()
            except Exception as e:
                w(f"  header={header_row}: ERROR - {e}")
    except Exception as e:
        w(f"  Could not read with pandas: {e}")

wb.close()

# Write to file
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print(f"\n\nOutput saved to: {OUTPUT}")
