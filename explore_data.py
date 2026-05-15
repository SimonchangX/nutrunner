import pandas as pd
import os

folder = r"C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]

print(f"Found {len(files)} Excel files\n")

# Analyze first file
first_file = os.path.join(folder, files[0])
print(f"Analyzing: {files[0]}\n")
print("=" * 80)

# Get sheet names
xls = pd.ExcelFile(first_file)
print(f"Sheet names: {xls.sheet_names}\n")

# Read each sheet
for sheet in xls.sheet_names:
    print(f"\n{'='*80}")
    print(f"SHEET: {sheet}")
    print(f"{'='*80}")
    
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string())
    
    print(f"\nData types:")
    print(df.dtypes.to_string())
    
    print("\n" + "=" * 80)
