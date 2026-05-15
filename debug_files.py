import glob, os, pandas as pd

folder = r"c:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
os.chdir(folder)

files = glob.glob(os.path.join(folder, "*.xlsx"))
print(f"Total glob finds: {len(files)} files\n")

for f in sorted(files):
    bn = os.path.basename(f)
    try:
        sheets = pd.ExcelFile(f).sheet_names
        has_target = "曲线 覆盖" in sheets
        print(f"  {bn}: sheets={sheets}, has_curve_cover={has_target}")
    except Exception as e:
        print(f"  {bn}: ERROR - {e}")
