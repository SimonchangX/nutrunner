"""
Atlas Copco ToolsNet8 - Advanced Trace Anomaly Detection System
Upgraded with physics-based features from original atlas_anomaly_system.py

Key enhancements:
- Joint stiffness analysis (dTorque/dAngle)
- 40+ physics-based features
- 3-layer ensemble scoring (Physics + Z-score + ML)
- Frequency domain analysis (FFT)
- Energy/work calculations
- Phase-based analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


class TraceAnomalyDetector:
    """Advanced anomaly detection for tightening traces with physics-based analysis"""
    
    def __init__(self, torque_limit_low=165, torque_limit_high=175, 
                 angle_limit_low=None, angle_limit_high=None):
        self.torque_limit_low = torque_limit_low
        self.torque_limit_high = torque_limit_high
        self.angle_limit_low = angle_limit_low
        self.angle_limit_high = angle_limit_high
        self.results_df = None
        self.anomalies_df = None
        
        # Physics-based thresholds (calibrated for Anti-roll bar 170Nm)
        # These can be adjusted based on your specific process knowledge
        self.physics_rules = {
            'max_stiffness_drop': -50.0,      # Nm/deg - indicates slip (calibrated from data)
            'min_stiffness_ratio': 0.01,       # Max/min stiffness ratio (relaxed)
            'max_overshoot': 0.25,            # 25% overshoot (relaxed)
            'max_high_freq_energy': 0.5,      # Too much vibration (relaxed)
            'min_linearity': 0.5,             # R² of linear fit (relaxed for real curves)
        }
        
        # Reference statistics for Z-score
        self.reference_medians = None
        self.reference_iqrs = None
        
        # Data-calibrated physics thresholds
        self.calibrated_physics_rules = None
        
    def calibrate_physics_rules_from_data(self, features_df):
        """Calibrate physics thresholds from data percentiles instead of hardcoded values.
        Clips against default physics_rules to avoid extreme values from mixed-program data.
        """
        self.calibrated_physics_rules = {}

        def _clip(val, lo, hi):
            return float(np.clip(val, lo, hi))

        # Stiffness drop: more negative is worse. Use more conservative (higher) of calibrated vs default.
        col = features_df.get('stiffness_drop_max', pd.Series([0]))
        p5 = np.percentile(col, 5)
        self.calibrated_physics_rules['stiffness_drop_thresh'] = _clip(
            p5, self.physics_rules['max_stiffness_drop'], 0)

        # Overshoot: higher is worse. Use more conservative (lower) of calibrated vs default.
        col = features_df.get('overshoot_ratio', pd.Series([0]))
        p95 = np.percentile(col, 99)  # use 99th to ignore extreme outliers
        self.calibrated_physics_rules['overshoot_thresh'] = _clip(
            p95, 0.01, self.physics_rules['max_overshoot'])

        # High freq energy: higher is worse
        col = features_df.get('high_freq_energy', pd.Series([0.5]))
        p95 = np.percentile(col, 99)
        self.calibrated_physics_rules['high_freq_thresh'] = _clip(
            p95, 0.01, self.physics_rules['max_high_freq_energy'])

        # Linearity: lower is worse. Use more conservative (higher) of calibrated vs default.
        col = features_df.get('shape_linearity', pd.Series([1]))
        p5 = np.percentile(col, 5)
        self.calibrated_physics_rules['linearity_thresh'] = _clip(
            p5, 0.01, self.physics_rules['min_linearity'])

        # Flat regions: higher is worse
        col = features_df.get('flat_regions_ratio', pd.Series([0]))
        p95 = np.percentile(col, 99)
        self.calibrated_physics_rules['flat_regions_thresh'] = _clip(
            p95, 0.01, 1.0)

        # Stiffness ratio: higher is worse
        col = features_df.get('stiffness_ratio', pd.Series([0]))
        p95 = np.percentile(col, 99)
        self.calibrated_physics_rules['stiffness_ratio_thresh'] = _clip(
            p95, 10, 200)

        print(f"  Calibrated: stiffness_drop<{self.calibrated_physics_rules['stiffness_drop_thresh']:.1f}, "
              f"overshoot>{self.calibrated_physics_rules['overshoot_thresh']:.3f}, "
              f"linearity<{self.calibrated_physics_rules['linearity_thresh']:.3f}, "
              f"flat>{self.calibrated_physics_rules['flat_regions_thresh']:.2f}")

    @staticmethod
    def estimate_contamination(scores, min_rate=0.02, max_rate=0.25):
        """Estimate contamination rate using elbow method on sorted scores"""
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        if n > 10:
            x = np.arange(n)
            y = sorted_scores
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-6)
            dists = np.abs(y_norm - x_norm)
            elbow_idx = np.argmax(dists)
            estimated = 1.0 - elbow_idx / n
        else:
            estimated = 0.10
        return float(np.clip(estimated, min_rate, max_rate))
    
    def load_data(self, file_paths):
        """Load and combine data from multiple Excel files"""
        all_data = []
        for file_path in file_paths:
            df = pd.read_excel(file_path, sheet_name='曲线 覆盖')
            df['source_file'] = file_path.split('\\')[-1]
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} samples from {len(file_paths)} files")
        print(f"Unique tightening results: {combined_df['结果 ID'].nunique()}")
        return combined_df
    
    def extract_traces(self, df):
        """Extract individual tightening traces from the combined data"""
        unique_results = df['结果 ID'].unique()
        traces = []

        for result_id in unique_results:
            result_data = df[df['结果 ID'] == result_id].copy()
            result_data = result_data.sort_values('采样时间')

            if len(result_data) == 0:
                continue

            trace_info = {
                'result_id': result_id,
                'source_file': result_data['source_file'].iloc[0],
                'program': result_data['程序'].iloc[0],
                'result_time': result_data['结果时间'].iloc[0],
                'final_torque': result_data['最终扭矩 (N·m)'].iloc[0],
                'final_angle': result_data['最终角度 (度)'].iloc[0],
                'max_torque': result_data['最大扭矩 (N·m)'].iloc[0],
                'max_angle': result_data['最大角度 (度)'].iloc[0],
                'torque_time_series': result_data['扭矩 (N·m)'].values,
                'angle_time_series': result_data['角度 (度)'].values,
                'current_time_series': result_data['电流 (A)'].values,
                'time_series': result_data['采样时间'].values,
                'samples_count': len(result_data)
            }

            traces.append(trace_info)

        self.results_df = pd.DataFrame(traces)
        return self.results_df
    
    def calculate_stiffness(self, torque, angle, window=5):
        """
        Calculate joint stiffness (dTorque/dAngle) with smoothing
        From original atlas_anomaly_system.py
        """
        if len(angle) < 2 or len(torque) < 2:
            return np.array([])
        
        d_angle = np.diff(angle)
        d_torque = np.diff(torque)
        
        # Avoid division by zero
        d_angle[d_angle == 0] = 1e-6
        
        stiffness = d_torque / d_angle
        
        # Smooth with moving average
        if len(stiffness) > window:
            window = min(window, len(stiffness) // 2 + 1)
            stiffness = np.convolve(stiffness, np.ones(window)/window, mode='same')
        
        return stiffness
    
    def extract_40_features(self, trace_row):
        """
        Extract 40+ physics-based features (from original system)
        Much more comprehensive than basic statistics
        """
        torque = trace_row['torque_time_series']
        angle = trace_row['angle_time_series']
        current = trace_row['current_time_series']
        time = trace_row['time_series']
        
        features = {}
        
        # ==================== 1. BASIC STATISTICS (10 features) ====================
        features['torque_mean'] = np.mean(torque)
        features['torque_std'] = np.std(torque)
        features['torque_max'] = np.max(torque)
        features['torque_min'] = np.min(torque)
        features['torque_range'] = np.max(torque) - np.min(torque)
        features['torque_cv'] = np.std(torque) / (np.mean(torque) + 1e-6)  # Coefficient of variation
        features['angle_total'] = angle[-1] - angle[0] if len(angle) > 0 else 0.0
        features['torque_final'] = torque[-1]
        features['curve_duration_ms'] = time[-1] - time[0] if len(time) > 0 else 0.0
        features['n_points'] = float(len(torque))
        
        # ==================== 2. CURVE SHAPE (5 features) ====================
        if len(torque) > 2:
            # Normalize for shape analysis
            t_norm = (torque - np.min(torque)) / (np.max(torque) - np.min(torque) + 1e-6)
            a_norm = np.linspace(0, 1, len(torque))
            
            # Linearity (R² of linear fit)
            slope, intercept = np.polyfit(a_norm, t_norm, 1)
            predicted = slope * a_norm + intercept
            ss_res = np.sum((t_norm - predicted) ** 2)
            ss_tot = np.sum((t_norm - np.mean(t_norm)) ** 2)
            features['shape_linearity'] = 1 - (ss_res / (ss_tot + 1e-6))
            
            # Skewness and kurtosis
            features['shape_skewness'] = float(np.mean(((torque - np.mean(torque)) / (np.std(torque) + 1e-6)) ** 3))
            features['shape_kurtosis'] = float(np.mean(((torque - np.mean(torque)) / (np.std(torque) + 1e-6)) ** 4))
            
            # Entropy of torque distribution
            hist, _ = np.histogram(t_norm, bins=10, range=(0, 1), density=True)
            hist = hist[hist > 0]
            features['shape_entropy'] = float(-np.sum(hist * np.log(hist + 1e-6)))
            
            # Midpoint ratio: where does 50% of final torque occur
            half_torque = torque[-1] * 0.5
            idx_half = np.searchsorted(torque, half_torque)
            features['midpoint_ratio'] = idx_half / len(torque) if len(torque) > 0 else 0.5
        else:
            features.update({'shape_linearity': 0, 'shape_skewness': 0, 'shape_kurtosis': 0, 
                           'shape_entropy': 0, 'midpoint_ratio': 0.5})
        
        # ==================== 3. STIFFNESS FEATURES (10 features) ====================
        stiffness = self.calculate_stiffness(torque, angle)
        
        if len(stiffness) > 2:
            # Split into phases
            n = len(stiffness)
            early = stiffness[:n//3]
            mid = stiffness[n//3:2*n//3]
            late = stiffness[2*n//3:]
            
            features['stiffness_mean'] = np.mean(stiffness)
            features['stiffness_std'] = np.std(stiffness)
            features['stiffness_max'] = np.max(stiffness)
            features['stiffness_min'] = np.min(stiffness)
            features['stiffness_ratio'] = np.max(stiffness) / (np.min(stiffness) + 1e-6)
            
            # Stiffness drops (slip detection)
            stiffness_diff = np.diff(stiffness)
            features['stiffness_drop_max'] = float(np.min(stiffness_diff))
            
            features['stiffness_final'] = stiffness[-1]
            features['stiffness_early'] = np.mean(early)
            features['stiffness_mid'] = np.mean(mid)
            features['stiffness_late'] = np.mean(late)
            features['stiffness_trend'] = float(np.polyfit(range(len(stiffness)), stiffness, 1)[0])
        else:
            for key in ['stiffness_mean', 'stiffness_std', 'stiffness_max', 'stiffness_min',
                       'stiffness_ratio', 'stiffness_drop_max', 'stiffness_final',
                       'stiffness_early', 'stiffness_mid', 'stiffness_late', 'stiffness_trend']:
                features[key] = 0.0
        
        # ==================== 4. DYNAMIC BEHAVIOR (7 features) ====================
        if len(torque) > 2:
            # Torque rate (Nm/ms)
            torque_rate = np.diff(torque) / np.diff(time + 1e-6)
            features['torque_rate_mean'] = np.mean(torque_rate)
            features['torque_rate_max'] = np.max(torque_rate)
            features['torque_rate_std'] = np.std(torque_rate)
            
            # Overshoot detection (capped at 1.0 to avoid extreme values from partial traces)
            max_idx = np.argmax(torque)
            if max_idx < len(torque) - 1:
                ratio = (torque[max_idx] - torque[-1]) / (torque[-1] + 1e-6)
                features['overshoot_ratio'] = float(np.clip(ratio, 0.0, 1.0))
            else:
                features['overshoot_ratio'] = 0.0
            
            # Settling time
            threshold = torque[-1] * 0.95
            settled_idx = np.where(torque >= threshold)[0]
            features['settling_time_ratio'] = settled_idx[0] / len(torque) if len(settled_idx) > 0 else 1.0
            
            features['velocity_mean'] = np.mean(np.diff(angle) / np.diff(time + 1e-6))
            features['velocity_std'] = np.std(np.diff(angle) / np.diff(time + 1e-6))
        else:
            for key in ['torque_rate_mean', 'torque_rate_max', 'torque_rate_std',
                       'overshoot_ratio', 'settling_time_ratio', 'velocity_mean', 'velocity_std']:
                features[key] = 0.0
        
        # ==================== 5. ENERGY & WORK (4 features) ====================
        if len(torque) > 1:
            # Work = integral of torque over angle
            features['work_total'] = np.trapezoid(torque, angle)
            
            # Theoretical work if linear
            work_theoretical = 0.5 * torque[-1] * (angle[-1] - angle[0])
            features['work_efficiency'] = features['work_total'] / (work_theoretical + 1e-6)
            
            features['energy_per_angle'] = features['work_total'] / (angle[-1] - angle[0] + 1e-6)
            features['torque_integral'] = np.trapezoid(torque, time)
        else:
            features.update({'work_total': 0, 'work_efficiency': 0, 
                           'energy_per_angle': 0, 'torque_integral': 0})
        
        # ==================== 6. FREQUENCY DOMAIN (4 features) ====================
        if len(torque) > 16:
            # Detrend
            t_detrended = torque - np.polyval(np.polyfit(range(len(torque)), torque, 1), range(len(torque)))
            
            # FFT
            fft_vals = rfft(t_detrended)
            fft_power = np.abs(fft_vals) ** 2
            freqs = rfftfreq(len(t_detrended))
            
            # Dominant frequency
            if len(fft_power) > 1:
                dominant_idx = np.argmax(fft_power[1:]) + 1
                features['dominant_freq'] = freqs[dominant_idx]
            else:
                features['dominant_freq'] = 0.0
            
            # High frequency energy
            high_freq_mask = freqs > 0.2 * 0.5  # >20% of Nyquist
            features['high_freq_energy'] = np.sum(fft_power[high_freq_mask]) / (np.sum(fft_power) + 1e-6)
            
            # Spectral entropy
            power_norm = fft_power / (np.sum(fft_power) + 1e-6)
            features['spectral_entropy'] = -np.sum(power_norm * np.log(power_norm + 1e-6))
            
            features['fft_max_power'] = np.max(fft_power)
        else:
            features.update({'dominant_freq': 0, 'high_freq_energy': 0, 
                           'spectral_entropy': 0, 'fft_max_power': 0})
        
        # ==================== 7. EARLY/LATE PHASE (6 features) ====================
        early_cutoff = int(len(torque) * 0.3)
        features['early_torque_mean'] = np.mean(torque[:early_cutoff])
        features['early_torque_std'] = np.std(torque[:early_cutoff])
        features['early_current_mean'] = np.mean(current[:early_cutoff])
        
        mid_start = int(len(torque) * 0.3)
        mid_end = int(len(torque) * 0.7)
        features['mid_torque_std'] = np.std(torque[mid_start:mid_end])
        
        late_start = int(len(torque) * 0.8)
        features['late_torque_std'] = np.std(torque[late_start:])
        features['late_torque_mean'] = np.mean(torque[late_start:])
        
        # Final stability
        final_10_pct = int(len(torque) * 0.9)
        features['torque_final_stability'] = np.std(torque[final_10_pct:])
        
        # ==================== 8. ANOMALY INDICATORS (4 features) ====================
        features['negative_torque_ratio'] = np.sum(torque < 0) / len(torque)
        features['torque_spike_count'] = np.sum(np.abs(np.diff(torque)) > 10)
        features['current_spike_count'] = np.sum(np.abs(np.diff(current)) > 5)
        
        # Flat regions (galled thread detection)
        if len(stiffness) > 10:
            flat_ratio = np.sum(np.abs(stiffness) < 0.1) / len(stiffness)
            features['flat_regions_ratio'] = flat_ratio
        else:
            features['flat_regions_ratio'] = 0.0
        
        return features
    
    def detect_anomalies_ensemble(self, contamination=0.1):
        """
        3-Layer Ensemble Detection (from original system)
        Layer 1: Physics-based rules (40%)
        Layer 2: Statistical Z-score (20%)
        Layer 3: Isolation Forest ML (40%)
        """
        print("Extracting 40+ physics-based features...")
        features_list = []
        for idx, row in self.results_df.iterrows():
            features = self.extract_40_features(row)
            features['result_id'] = row['result_id']
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        self._features_df = features_df  # Store for later visualization (PCA)

        # Calibrate physics thresholds from this dataset
        self.calibrate_physics_rules_from_data(features_df)

        # Prepare features for ML
        feature_cols = [col for col in features_df.columns if col != 'result_id']
        X = features_df[feature_cols]
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        print("Running 3-layer ensemble detection...")

        # ==================== LAYER 1: Physics-based scoring (40%) ====================
        print("  Layer 1: Physics-based rules...")
        physics_scores = np.zeros(len(X))
        physics_violations = []

        for i, (_, row) in enumerate(features_df.iterrows()):
            violations = []
            score = 0.0

            pr = self.calibrated_physics_rules if self.calibrated_physics_rules else self.physics_rules

            thresh = pr.get('stiffness_drop_thresh', self.physics_rules['max_stiffness_drop'])
            if row.get('stiffness_drop_max', 0) < thresh:
                violations.append("stiffness_drop")
                score += 0.2

            thresh = pr.get('overshoot_thresh', self.physics_rules['max_overshoot'])
            if row.get('overshoot_ratio', 0) > thresh:
                violations.append("overshoot")
                score += 0.2

            thresh = pr.get('high_freq_thresh', self.physics_rules['max_high_freq_energy'])
            if row.get('high_freq_energy', 0) > thresh:
                violations.append("high_vibration")
                score += 0.15

            thresh = pr.get('linearity_thresh', self.physics_rules['min_linearity'])
            if row.get('shape_linearity', 1.0) < thresh:
                violations.append("non_linear")
                score += 0.15

            thresh = pr.get('flat_regions_thresh', 0.3)
            if row.get('flat_regions_ratio', 0) > thresh:
                violations.append("flat_regions")
                score += 0.15

            thresh = pr.get('stiffness_ratio_thresh', 100)
            if row.get('stiffness_ratio', 0) > thresh:
                violations.append("stiffness_ratio")
                score += 0.15

            physics_scores[i] = min(1.0, score)
            physics_violations.append(violations)

        # ==================== LAYER 2: Statistical Z-score (20%) ====================
        print("  Layer 2: Statistical Z-score (RobustScaler + rank)...")
        zscore_keys = ['torque_mean', 'torque_std', 'stiffness_mean', 'stiffness_ratio',
                       'work_total', 'torque_cv', 'shape_linearity', 'overshoot_ratio',
                       'high_freq_energy', 'settling_time_ratio']

        zscore_matrix = features_df[zscore_keys].values
        zscore_matrix = np.nan_to_num(zscore_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

        # Use RobustScaler (median + IQR) instead of raw MAD to avoid z-score explosion
        robust_scaler = RobustScaler(quantile_range=(10, 90))
        zscore_scaled = robust_scaler.fit_transform(zscore_matrix)

        # Euclidean distance from origin — 0 means all features at median
        zscore_dist = np.sqrt(np.sum(zscore_scaled ** 2, axis=1))

        # Convert to 0-1 using percentile rank (adaptive, no magic thresholds)
        z_scores_norm = rankdata(zscore_dist) / len(zscore_dist)

        # Track which dimensions contributed (|scaled| > 3 IQR from median)
        zscore_keys_arr = np.array(zscore_keys)
        zscore_reasons = []
        dim_labels = {
            'torque_mean': 'torque_level', 'torque_std': 'torque_variation',
            'stiffness_mean': 'stiffness_level', 'stiffness_ratio': 'stiffness_variation',
            'work_total': 'work_energy', 'torque_cv': 'torque_instability',
            'shape_linearity': 'curve_shape', 'overshoot_ratio': 'overshoot',
            'high_freq_energy': 'vibration', 'settling_time_ratio': 'settling_time'
        }
        for i in range(len(zscore_matrix)):
            dims = zscore_keys_arr[np.where(np.abs(zscore_scaled[i]) > 3.0)[0]]
            reason_labels = []
            for d in dims:
                if d in dim_labels and dim_labels[d] not in reason_labels:
                    reason_labels.append(dim_labels[d])
            zscore_reasons.append(reason_labels)

        # ==================== LAYER 3: Isolation Forest ML (40%) ====================
        print("  Layer 3: Isolation Forest with 40+ physics features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=300,
            max_samples=256
        )

        predictions = iso_forest.fit_predict(X_scaled)
        scores = -iso_forest.score_samples(X_scaled)  # Convert to positive anomaly scores

        # Normalize ML scores to 0-1 using percentile rank (data-driven)
        ml_scores_norm = rankdata(scores) / len(scores)

        # ==================== ENSEMBLE: Weighted combination ====================
        ensemble_score = (
            0.4 * physics_scores +
            0.2 * z_scores_norm +
            0.4 * ml_scores_norm
        )

        # Adaptive threshold on ensemble score instead of raw IF classification
        threshold = np.percentile(ensemble_score, 100 * (1 - contamination))
        is_anomaly = ensemble_score >= threshold

        # Combine all reasons
        all_reasons = []
        for i in range(len(X)):
            reasons = []
            if physics_violations[i]:
                reasons.extend(physics_violations[i])
            if zscore_reasons[i]:
                reasons.extend(zscore_reasons[i])
            if predictions[i] == -1:
                if not reasons:
                    reasons.append('ml_pattern_anomaly')
                else:
                    reasons.append('ml_confirmed')
            all_reasons.append('; '.join(reasons) if reasons else 'normal')

        # Build results
        features_df['physics_score'] = physics_scores
        features_df['zscore_score'] = z_scores_norm
        features_df['ml_score'] = ml_scores_norm
        features_df['anomaly_score'] = ensemble_score
        features_df['is_anomaly'] = is_anomaly
        features_df['anomaly_reasons'] = all_reasons

        # Merge with results
        self.anomalies_df = self.results_df.merge(
            features_df[['result_id', 'anomaly_score', 'is_anomaly',
                        'anomaly_reasons', 'physics_score', 'zscore_score', 'ml_score']],
            on='result_id'
        )

        anomaly_count = self.anomalies_df['is_anomaly'].sum()
        print(f"Detection complete: {anomaly_count} anomalies found")

        return self.anomalies_df
    
    def plot_trace_comparison(self, result_ids=None, max_anomalies=10):
        """Plot torque and angle traces, highlighting anomalies"""
        if result_ids is None:
            if self.anomalies_df is None or 'is_anomaly' not in self.anomalies_df.columns:
                # Fallback: just show first few traces from results_df
                if self.results_df is not None and len(self.results_df) > 0:
                    result_ids = self.results_df['result_id'].head(max_anomalies).tolist()
                else:
                    fig = go.Figure()
                    fig.add_annotation(text="No traces available to display", showarrow=False, font=dict(size=20))
                    return fig
            else:
                anomalies = self.anomalies_df[self.anomalies_df['is_anomaly']].head(max_anomalies)
                normals = self.anomalies_df[~self.anomalies_df['is_anomaly']].head(max_anomalies)
                selected = pd.concat([anomalies, normals])
                result_ids = selected['result_id'].tolist()
        
        if not result_ids:
            fig = go.Figure()
            fig.add_annotation(text="No traces to display", showarrow=False, font=dict(size=20))
            return fig
        
        # Create a lookup dict for faster access
        traces_dict = {row['result_id']: row for _, row in self.anomalies_df.iterrows()}
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Torque vs Time', 'Angle vs Time', 'Stiffness (dTorque/dAngle)'),
            vertical_spacing=0.1
        )
        
        for result_id in result_ids:
            if result_id not in traces_dict:
                continue

            trace_data = traces_dict[result_id]
            is_anomaly = trace_data.get('is_anomaly', False)
            
            # Torque
            fig.add_trace(
                go.Scatter(
                    x=trace_data['time_series'],
                    y=trace_data['torque_time_series'],
                    mode='lines',
                    name=f"ID {result_id} {'⚠️' if is_anomaly else '✓'}",
                    line=dict(width=2, color='red' if is_anomaly else 'blue')
                ),
                row=1, col=1
            )
            
            # Angle
            fig.add_trace(
                go.Scatter(
                    x=trace_data['time_series'],
                    y=trace_data['angle_time_series'],
                    mode='lines',
                    name=f"ID {result_id}",
                    line=dict(width=2, color='red' if is_anomaly else 'blue'),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Stiffness
            stiffness = self.calculate_stiffness(
                trace_data['torque_time_series'],
                trace_data['angle_time_series']
            )
            time_stiffness = trace_data['time_series'][:-1] if len(trace_data['time_series']) > 1 else trace_data['time_series']
            
            fig.add_trace(
                go.Scatter(
                    x=time_stiffness,
                    y=stiffness,
                    mode='lines',
                    name=f"ID {result_id} Stiffness",
                    line=dict(width=2, color='red' if is_anomaly else 'blue'),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(height=1000, title_text="Torque, Angle, and Stiffness Comparison")
        fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
        fig.update_yaxes(title_text="Torque (N·m)", row=1, col=1)
        fig.update_yaxes(title_text="Angle (deg)", row=2, col=1)
        fig.update_yaxes(title_text="Stiffness (N·m/deg)", row=3, col=1)
        
        return fig
    
    def plot_single_trace_detail(self, result_id):
        """Detailed view of a single trace with all physics features"""
        # Use dict lookup for faster access
        traces_dict = {row['result_id']: row for _, row in self.anomalies_df.iterrows()}
        
        if result_id not in traces_dict:
            fig = go.Figure()
            fig.add_annotation(text=f"Trace {result_id} not found", showarrow=False, font=dict(size=20))
            return fig
        
        trace_data = traces_dict[result_id]
        
        torque = trace_data['torque_time_series']
        angle = trace_data['angle_time_series']
        current = trace_data['current_time_series']
        time = trace_data['time_series']
        
        # Calculate stiffness
        stiffness = self.calculate_stiffness(torque, angle)
        time_stiffness = time[:-1] if len(time) > 1 else time
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Torque', 'Angle', 'Stiffness', 'Current'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Torque with limits
        fig.add_trace(
            go.Scatter(x=time, y=torque, mode='lines', name='Torque', line=dict(width=2)),
            row=1, col=1
        )
        fig.add_hline(y=self.torque_limit_low, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=self.torque_limit_high, line_dash="dash", line_color="red", row=1, col=1)
        
        # Angle
        fig.add_trace(
            go.Scatter(x=time, y=angle, mode='lines', name='Angle', line=dict(width=2, color='green')),
            row=2, col=1
        )
        
        # Stiffness (NEW!)
        fig.add_trace(
            go.Scatter(x=time_stiffness, y=stiffness, mode='lines', name='Stiffness', 
                      line=dict(width=2, color='purple')),
            row=3, col=1
        )
        
        # Current
        fig.add_trace(
            go.Scatter(x=time, y=current, mode='lines', name='Current', 
                      line=dict(width=2, color='orange')),
            row=4, col=1
        )
        
        fig.update_layout(
            height=1200,
            title_text=f"Complete Trace Analysis - ID {result_id} {'⚠️ ANOMALY' if trace_data.get('is_anomaly', False) else '✓ NORMAL'}"
        )
        fig.update_xaxes(title_text="Time (ms)", row=4, col=1)
        fig.update_yaxes(title_text="Torque (N·m)", row=1, col=1)
        fig.update_yaxes(title_text="Angle (deg)", row=2, col=1)
        fig.update_yaxes(title_text="Stiffness (N·m/deg)", row=3, col=1)
        fig.update_yaxes(title_text="Current (A)", row=4, col=1)
        
        return fig
    
    def plot_pca_anomalies(self):
        """Project 40+ features to 2D PCA for visual anomaly inspection"""
        if self.anomalies_df is None or not hasattr(self, '_features_df'):
            return None

        # Only select numeric columns — exclude result_id, anomaly_reasons, etc.
        numeric_df = self._features_df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c != 'result_id']
        if len(feature_cols) < 2:
            return None

        X = numeric_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        is_anomaly_arr = self.anomalies_df['is_anomaly'].values
        trace_ids = self.anomalies_df['result_id'].tolist()

        fig = go.Figure()
        for label, color, symbol, name in [
            (False, 'steelblue', 'circle', 'Normal'),
            (True, 'red', 'x', 'Anomaly')
        ]:
            mask = is_anomaly_arr == label
            if mask.sum() == 0:
                continue
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode='markers',
                marker=dict(size=6, color=color, symbol=symbol),
                name=name,
                text=[f"ID: {tid}" for tid in np.array(trace_ids)[mask]],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))

        var_ratio = pca.explained_variance_ratio_
        fig.update_layout(
            title=f'PCA Projection of Traces ({var_ratio.sum():.1%} variance explained)',
            xaxis_title=f'PC1 ({var_ratio[0]:.1%})',
            yaxis_title=f'PC2 ({var_ratio[1]:.1%})',
            height=600
        )
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics of the analysis"""
        if self.anomalies_df is None:
            return None
        
        total = len(self.anomalies_df)
        anomaly_count = int(self.anomalies_df['is_anomaly'].sum())
        normal_count = total - anomaly_count
        
        return {
            'total_traces': total,
            'normal_traces': normal_count,
            'anomalous_traces': anomaly_count,
            'anomaly_rate': anomaly_count / total * 100,
            'anomaly_ids': self.anomalies_df[self.anomalies_df['is_anomaly']]['result_id'].tolist()
        }
