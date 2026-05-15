# 🔧 Atlas Copco ToolsNet8 - Trace Anomaly Detection System

This system analyzes tightening curve data from Atlas Copco ToolsNet8 to detect abnormal traces, even when final torque and angle values are within specifications.

## 🎯 What It Detects

The system can identify various types of anomalies:

1. **Negative Torque Spikes** - Unexpected reverse torque during tightening
2. **Torque Drops** - Sudden decreases in torque (potential slip or backlash)
3. **High Early Variance** - Unstable behavior in the initial phase
4. **Unstable Final Torque** - Torque not stabilizing properly at the end
5. **Current Anomalies** - Abnormal motor current patterns
6. **Angle Reversals** - Angle going backward during tightening
7. **Statistical Anomalies** - Patterns detected by Machine Learning (Isolation Forest)

## 📁 Data Structure

The system expects Excel files from ToolsNet8 with the "曲线 覆盖" (Curve Coverage) sheet containing:
- Time series data (采样时间)
- Torque values (扭矩)
- Angle values (角度)
- Current values (电流)
- Final results (最终扭矩, 最终角度)

## 🚀 Quick Start

### 1. Install Dependencies (if needed)
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ streamlit plotly pandas numpy scikit-learn
```

### 2. Run the Dashboard App
```bash
cd "C:\Users\SZHH59\PyCharmMiscProject\Nutrunner\Anti-roll bar"
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### Overview Tab
- Data statistics and distributions
- Final torque and angle histograms
- Sample count analysis

### Anomaly Detection Tab
- Run anomaly detection with adjustable sensitivity
- View detected anomalies with reasons
- Anomaly scoring and classification

### Trace Viewer Tab
- **Single Trace**: Detailed view of one tightening cycle
- **Compare Traces**: Side-by-side comparison of normal vs anomalous
- **All Anomalies**: View all detected anomalies together

### Export Results Tab
- Download results as CSV
- Summary statistics
- Anomaly reports

## 🔍 How It Works

### 1. Isolation Forest (Machine Learning)
- Extracts 24 statistical features from each trace
- Detects outliers based on multivariate analysis
- Adjustable contamination rate (expected anomaly percentage)

### 2. Rule-Based Detection
- Negative torque detection
- Torque spike detection (>20 N·m sudden change)
- Early phase variance analysis
- Final torque stability check
- Current pattern analysis
- Angle reversal detection

## 📈 Features Extracted

The system analyzes:
- Torque statistics (mean, std, skewness, kurtosis)
- Angle progression and slope
- Current patterns
- Rate of change metrics
- Early/middle/late phase analysis
- Torque stability metrics
- Spike and drop detection

## ⚙️ Configuration

You can adjust:
- **Contamination Rate**: Expected percentage of anomalies (1-50%)
- **Torque Limits**: High/low torque thresholds
- **Detection Sensitivity**: Through the slider in the UI

## 📋 Output

The system provides:
- Anomaly score for each trace
- Binary classification (Normal/Anomalous)
- Detailed reasons for each anomaly
- Interactive visualizations
- Exportable CSV reports

## 🐛 Troubleshooting

**App won't start:**
```bash
streamlit run app.py
```

**No data found:**
- Ensure Excel files are in the specified folder
- Check that files contain "曲线 覆盖" sheet

**Slow performance:**
- Reduce the number of files loaded
- Increase contamination rate to reduce false positives

## 📝 Example Use Case

For Anti-roll bar tightening:
- Target: 170 N·m
- Files: Anti roll bar-1.xlsx through Anti roll bar-7.xlsx
- Results: ~350+ tightening cycles analyzed
- Detection: Identifies traces with abnormal patterns despite acceptable final values

## 📞 Support

For issues or questions, check:
1. Console output for error messages
2. Ensure all dependencies are installed
3. Verify data format matches ToolsNet8 export format

---

**Built with:** Streamlit, Plotly, Scikit-learn, Pandas, NumPy
