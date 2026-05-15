import os
import pandas as pd
from model_manager import load_all_traces
from anomaly_detector import TraceAnomalyDetector

folder = r"C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"

detector = TraceAnomalyDetector()
combined, files = load_all_traces(folder, use_cache=True)
df = detector.extract_traces(combined)

print("Running detection...")
anomalies = detector.detect_anomalies_ensemble(contamination=0.1)

print("\nScore Distribution:")
cols = ['result_id', 'anomaly_score', 'physics_score', 'zscore_score',
        'ml_score', 'is_anomaly']
available = [c for c in cols if c in anomalies.columns]
print(anomalies[available].head(20).to_string())

print(f"\nScore Statistics:")
for col, label in [('anomaly_score', 'Ensemble'), ('physics_score', 'Physics'),
                   ('zscore_score', 'Z-Score'), ('ml_score', 'ML')]:
    if col in anomalies.columns:
        s = anomalies[col]
        print(f"{label} Score: {s.min():.3f} - {s.max():.3f} (mean: {s.mean():.3f})")
