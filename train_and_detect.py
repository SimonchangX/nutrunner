"""
Anti-roll Bar Anomaly Detection - Training Script
Loads ALL Excel files from Anti-roll bar folder and runs the 3-layer ensemble detection.
Saves the trained model for future use.
"""

import os
import glob
import pandas as pd
import numpy as np
import pickle
from anomaly_detector import TraceAnomalyDetector

# ============================================================
# Configuration
# ============================================================
DATA_FOLDER = r"C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
MODEL_PATH = os.path.join(DATA_FOLDER, "antiroll_anomaly_model.pkl")
RESULTS_PATH = os.path.join(DATA_FOLDER, "anomaly_detection_results.csv")

# ============================================================
# Step 1: Load all data files
# ============================================================
def load_all_files(folder_path):
    """Load all Excel files from the Anti-roll bar folder"""
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not files:
        print(f"ERROR: No Excel files found in {folder_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Found {len(files)} Excel files:")
    for f in sorted(files):
        basename = os.path.basename(f)
        print(f"  - {basename}")
    print(f"{'='*60}\n")
    
    all_data = []
    for file_path in files:
        try:
            df = pd.read_excel(file_path, sheet_name='曲线 覆盖')
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)
            print(f"[OK] Loaded {len(df)} samples from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[ERR] Error loading {file_path}: {e}")
    
    if not all_data:
        print("ERROR: No data could be loaded!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n[DATA] Total: {len(combined_df):,} samples from {combined_df['结果 ID'].nunique()} tightening results")
    
    return combined_df


# ============================================================
# Step 2: Run anomaly detection
# ============================================================
def run_detection(combined_df, contamination=0.10):
    """Run the 3-layer ensemble anomaly detection"""
    print(f"\n{'='*60}")
    print("INITIALIZING ANOMALY DETECTOR")
    print(f"{'='*60}")
    
    detector = TraceAnomalyDetector()
    
    print("\nExtracting traces...")
    detector.results_df = detector.extract_traces(combined_df)
    print(f"[OK] Extracted {len(detector.results_df)} unique traces")
    
    print(f"\nRunning detection with contamination={contamination:.0%}...")
    anomalies_df = detector.detect_anomalies_ensemble(contamination=contamination)
    
    return detector, anomalies_df


# ============================================================
# Step 3: Save model and results
# ============================================================
def save_model(detector, anomalies_df, model_path, results_path):
    """Save the trained detector and results"""
    model_data = {
        'detector': detector,
        'reference_medians': detector.reference_medians,
        'reference_iqrs': detector.reference_iqrs,
        'physics_rules': detector.physics_rules,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n[SAVED] Model: {model_path}")
    
    # Save results
    export_cols = [
        'result_id', 'source_file', 'program', 'result_time',
        'final_torque', 'final_angle', 'max_torque', 'max_angle',
        'is_anomaly', 'anomaly_score', 'physics_score', 'zscore_score',
        'ml_score', 'anomaly_reasons'
    ]
    
    available_cols = [c for c in export_cols if c in anomalies_df.columns]
    anomalies_df[available_cols].to_csv(results_path, index=False)
    print(f"[SAVED] Results: {results_path}")


# ============================================================
# Step 4: Print summary
# ============================================================
def print_summary(anomalies_df):
    """Print a detailed summary of detection results"""
    print(f"\n{'='*60}")
    print("DETECTION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    total = len(anomalies_df)
    anomaly_count = anomalies_df['is_anomaly'].sum()
    normal_count = total - anomaly_count
    
    print(f"\n[DATA] Total Traces Analyzed: {total}")
    print(f"[OK] Normal Traces: {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"[WARN] Anomalous Traces: {anomaly_count} ({anomaly_count/total*100:.1f}%)")
    
    # Score distributions
    print("-" * 40)
    print("Score Distributions (0=normal, 1=highly anomalous):")
    print("-" * 40)
    print(f"  Physics Score (40%):  mean={anomalies_df['physics_score'].mean():.3f}, "
          f"max={anomalies_df['physics_score'].max():.3f}")
    print(f"  Z-Score (20%):        mean={anomalies_df['zscore_score'].mean():.3f}, "
          f"max={anomalies_df['zscore_score'].max():.3f}")
    print(f"  ML Score (40%):       mean={anomalies_df['ml_score'].mean():.3f}, "
          f"max={anomalies_df['ml_score'].max():.3f}")
    print(f"  Total Score:          mean={anomalies_df['anomaly_score'].mean():.3f}, "
          f"max={anomalies_df['anomaly_score'].max():.3f}")
    
    # Anomaly reasons
    if anomaly_count > 0:
        print(f"\n{'-'*40}")
        print("Top Anomaly Reasons:")
        print(f"{'-'*40}")
        
        anomalous = anomalies_df[anomalies_df['is_anomaly']]
        all_reasons = '; '.join(anomalous['anomaly_reasons'].dropna().tolist())
        reason_counts = {}
        for reason in all_reasons.replace(';', ',').split(','):
            reason = reason.strip()
            if reason and reason != 'normal':
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {reason}: {count}")
        
        # Show top anomalies
        print(f"\n{'-'*40}")
        print("Top 10 Most Anomalous Traces:")
        print(f"{'-'*40}")
        
        top_anomalies = anomalous.nlargest(10, 'anomaly_score')
        for _, row in top_anomalies.iterrows():
            print(f"  ID {row['result_id']:>8} | Score: {row['anomaly_score']:.3f} | "
                  f"File: {row.get('source_file', 'N/A')} | "
                  f"Torque: {row['final_torque']:.1f} Nm | "
                  f"Reasons: {row['anomaly_reasons'][:80]}")
    
    # Source file distribution
    print(f"\n{'-'*40}")
    print("Anomalies per Source File:")
    print(f"{'-'*40}")
    
    file_stats = anomalies_df.groupby('source_file').agg(
        total=('result_id', 'count'),
        anomalies=('is_anomaly', 'sum')
    ).reset_index()
    
    for _, row in file_stats.iterrows():
        bar = '#' * int(row['anomalies']) + '.' * int(row['total'] - row['anomalies'])
        print(f"  {row['source_file']:<50} {int(row['anomalies'])}/{int(row['total'])} {bar}")


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANTI-ROLL BAR ANOMALY DETECTION - TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n[LOAD] Loading data...")
    combined_df = load_all_files(DATA_FOLDER)
    
    if combined_df is None:
        print("\n[ERR] Failed to load data. Exiting.")
        exit(1)
    
    # Run detection
    contamination = 0.10  # Expected 10% anomaly rate
    detector, anomalies_df = run_detection(combined_df, contamination=contamination)
    
    # Save model and results
    save_model(detector, anomalies_df, MODEL_PATH, RESULTS_PATH)
    
    # Print summary
    print_summary(anomalies_df)
    
    print(f"\n{'='*60}")
    print("[DONE] TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Review results: {RESULTS_PATH}")
    print(f"  2. Run dashboard: streamlit run app.py")
    print(f"  3. Load saved model in future sessions from: {MODEL_PATH}")
