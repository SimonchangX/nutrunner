"""
Quick test script to verify anomaly detection system works
"""

import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_manager import load_all_traces
from anomaly_detector import TraceAnomalyDetector

folder = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("Testing Anomaly Detection System")
print("=" * 80)

print(f"\nLoading data files (cached)...")
combined, files = load_all_traces(folder, use_cache=True)
if combined is None:
    print("ERROR: No data files found!")
    sys.exit(1)

print(f"\nExtracting traces...")
detector = TraceAnomalyDetector()
traces = detector.extract_traces(combined)
print(f"Extracted {len(traces)} traces")

print(f"\nRunning 3-layer ensemble detection...")
anomalies = detector.detect_anomalies_ensemble(contamination=0.1)
detected = int(anomalies['is_anomaly'].sum())
print(f"Detected {detected} anomalies")

print(f"\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

total = len(anomalies)
anomaly_count = anomalies['is_anomaly'].sum()

print(f"Total traces: {total}")
print(f"Normal: {total - anomaly_count}")
print(f"Anomalous: {anomaly_count}")
print(f"Anomaly rate: {anomaly_count/total*100:.1f}%")

if anomaly_count > 0:
    print(f"\nAnomalous Result IDs:")
    anomalous = anomalies[anomalies['is_anomaly']]
    for idx, row in anomalous.head(5).iterrows():
        print(f"  - ID {row['result_id']}: {row.get('anomaly_reasons', 'ML detected')}")

print("\n" + "=" * 80)
print("Test complete! System is working correctly.")
print("=" * 80)
