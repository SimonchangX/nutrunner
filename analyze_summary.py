import pandas as pd
import os

folder = r"C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]

print(f"Analyzing all {len(files)} files\n")
print("=" * 80)

for file in sorted(files):
    filepath = os.path.join(folder, file)
    df = pd.read_excel(filepath, sheet_name='曲线 覆盖')
    
    unique_results = df['结果 ID'].nunique()
    total_rows = len(df)
    time_range = f"{df['采样时间'].min():.1f} to {df['采样时间'].max():.1f}"
    
    print(f"\n{file}:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique tightening results: {unique_results}")
    print(f"  Average samples per result: {total_rows/unique_results:.0f}")
    print(f"  Time range (ms): {time_range}")
    print(f"  Programs: {df['程序'].unique()}")
    print(f"  Final torque range: {df['最终扭矩 (N·m)'].min():.2f} - {df['最终扭矩 (N·m)'].max():.2f} N·m")
    print(f"  Final angle range: {df['最终角度 (度)'].min():.2f} - {df['最终角度 (度)'].max():.2f} deg")
