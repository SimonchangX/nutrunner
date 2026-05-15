"""
Model Manager - Handles automatic retraining and model loading
Used by both app.py (dashboard) and train_and_detect.py (standalone)
"""

import os
import glob
import pickle
import hashlib
import subprocess
import pandas as pd
import numpy as np
from anomaly_detector import TraceAnomalyDetector


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "antiroll_anomaly_model.pkl"
)
METADATA_PATH = MODEL_PATH.replace(".pkl", ".meta.json")
DATA_CACHE_PATH = MODEL_PATH.replace("antiroll_anomaly_model.pkl", "combined_data_cache.pkl")


def _get_folder_state(folder_path):
    """Get a fingerprint of all xlsx files in folder (names + sizes + mtimes)"""
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    state = []
    for f in files:
        stat = os.stat(f)
        state.append((os.path.basename(f), stat.st_size, stat.st_mtime))
    return state


def get_folder_fingerprint(folder_path):
    """Get a hash fingerprint of folder state"""
    state = _get_folder_state(folder_path)
    return hashlib.md5(str(state).encode()).hexdigest(), state


def load_saved_metadata():
    """Load saved model metadata (file list and fingerprint used during training)"""
    if os.path.exists(METADATA_PATH):
        import json
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_metadata(fingerprint, file_list, trace_count):
    """Save model training metadata"""
    import json
    meta = {
        "fingerprint": fingerprint,
        "files": [os.path.basename(f) for f in file_list],
        "file_count": len(file_list),
        "trace_count": trace_count,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def check_for_new_data(folder_path):
    """
    Check if there are new/changed/removed files since last training.
    Returns: (has_changes, current_fingerprint, saved_fingerprint, new_files, removed_files)
    """
    current_fp, current_state = get_folder_fingerprint(folder_path)
    meta = load_saved_metadata()

    if meta is None:
        # No previous training metadata
        return True, current_fp, None, [f[0] for f in current_state], []

    saved_fp = meta.get("fingerprint")
    saved_files = set(meta.get("files", []))

    current_files = set(f[0] for f in current_state)
    new_files = sorted(current_files - saved_files)
    removed_files = sorted(saved_files - current_files)

    has_changes = (current_fp != saved_fp) or len(new_files) > 0 or len(removed_files) > 0
    return has_changes, current_fp, saved_fp, new_files, removed_files


def _find_curve_sheet(excel_file):
    """Find the curve data sheet, trying common names."""
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names
    xls.close()

    # Try known sheet names in order of preference
    preferred = ['曲线 覆盖', '曲线', 'Curve', 'Data', 'Sheet1']
    for name in preferred:
        if name in sheet_names:
            return name
    # Fall back to first sheet
    if sheet_names:
        return sheet_names[0]
    return None


def _get_cache_fingerprint():
    """Get the fingerprint stored with the cache"""
    import json
    meta_path = DATA_CACHE_PATH + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f).get("fingerprint")
    return None

def _save_data_cache(combined, loaded_files, fingerprint):
    """Save combined dataframe to pickle cache with fingerprint"""
    cache_data = {
        "fingerprint": fingerprint,
        "combined": combined,
        "files": [os.path.basename(f) for f in loaded_files],
    }
    with open(DATA_CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f)
    import json
    with open(DATA_CACHE_PATH + ".meta", "w") as f:
        json.dump({"fingerprint": fingerprint}, f)


def load_all_traces(folder_path, use_cache=True):
    """Load all traces from xlsx files in folder (with optional pickle cache)"""
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    if not files:
        return None, []

    # Check cache first
    if use_cache and os.path.exists(DATA_CACHE_PATH):
        cached_fp = _get_cache_fingerprint()
        current_fp, _ = get_folder_fingerprint(folder_path)
        if cached_fp == current_fp:
            try:
                with open(DATA_CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                print(f"Loaded {len(cached['combined'])} samples from cache ({len(cached['files'])} files)")
                return cached["combined"], [os.path.join(folder_path, f) for f in cached["files"]]
            except Exception:
                pass  # Cache invalid, fall through to full load

    print("Reading Excel files (this may take a few minutes)...")
    all_data = []
    loaded_files = []
    for file_path in sorted(files):
        try:
            sheet_name = _find_curve_sheet(file_path)
            if sheet_name is None:
                continue
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df["source_file"] = os.path.basename(file_path)
            all_data.append(df)
            loaded_files.append(file_path)
            print(f"  Loaded {os.path.basename(file_path)} ({len(df)} rows)")
        except Exception as e:
            print(f"  Skipped {os.path.basename(file_path)}: {e}")
            continue

    if not all_data:
        return None, []

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total: {len(combined)} samples from {len(loaded_files)} files")

    # Save to cache for next time
    fp, _ = get_folder_fingerprint(folder_path)
    _save_data_cache(combined, loaded_files, fp)
    print(f"Saved cache for fast reload")

    return combined, loaded_files


def train_model(folder_path, contamination=0.10, callback=None):
    """
    Run full training pipeline.
    
    Args:
        folder_path: Path to data folder
        contamination: Expected anomaly rate
        callback: Optional function(status_msg) for progress updates
    
    def callback(msg):
        print(msg)
        st.info(msg)  # if in Streamlit context
    
    Returns:
        (detector, anomalies_df, info_dict) or None on failure
    """
    def log(msg):
        if callback:
            callback(msg)

    log("[Scanning for data files...]")
    combined, files = load_all_traces(folder_path)

    if combined is None or len(files) == 0:
        log("[ERROR] No valid data files found!")
        return None

    log(f"[DATA] Found {len(files)} files, {len(combined):,} samples")

    detector = TraceAnomalyDetector()
    detector.results_df = detector.extract_traces(combined)
    trace_count = len(detector.results_df)
    log(f"[TRACES] Extracted {trace_count} unique traces")

    log(f"[TRAINING] Running 3-layer ensemble detection (contamination={contamination:.0%})...")
    anomalies_df = detector.detect_anomalies_ensemble(contamination=contamination)

    # Save model
    model_data = {
        "detector": detector,
        "calibrated_physics_rules": detector.calibrated_physics_rules,
        "_features_df": getattr(detector, '_features_df', None),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    # Save results CSV
    results_path = os.path.join(folder_path, "anomaly_detection_results.csv")
    export_cols = [
        "result_id", "source_file", "program", "result_time",
        "final_torque", "final_angle", "max_torque", "max_angle",
        "is_anomaly", "anomaly_score", "physics_score", "zscore_score",
        "ml_score", "anomaly_reasons",
    ]
    available_cols = [c for c in export_cols if c in anomalies_df.columns]
    anomalies_df[available_cols].to_csv(results_path, index=False)

    # Save metadata
    fingerprint, state = get_folder_fingerprint(folder_path)
    save_metadata(fingerprint, files, trace_count)

    anomaly_count = int(anomalies_df["is_anomaly"].sum())
    log(f"[DONE] Training complete: {anomaly_count} anomalies in {trace_count} traces")
    log(f"[SAVED] Model: {os.path.basename(MODEL_PATH)}")

    return detector, anomalies_df, {
        "file_count": len(files),
        "trace_count": trace_count,
        "anomaly_count": anomaly_count,
        "fingerprint": fingerprint,
    }


def load_model():
    """Load previously saved model"""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
