# 🌲 Anomaly Detection using Isolation Forest
### AWS Cloud Metrics — Time-Series Anomaly Detection Pipeline

A full end-to-end machine learning pipeline that detects anomalies across **17 AWS infrastructure time-series datasets** (EC2, RDS, ELB, Network) using **Isolation Forest** with feature engineering, hyperparameter tuning, cross-validation, and rich visualizations.

---

## 📁 Project Structure

```
anomaly-detection-isolation-tree/
│
├── isolation_tree.ipynb          # Main pipeline notebook (6 steps)
│
├── data/                         # AWS time-series CSVs (place all CSVs here)
│   ├── ec2_cpu_utilization_5f5533.csv
│   ├── ec2_cpu_utilization_24ae8d.csv
│   ├── ec2_cpu_utilization_53ea38.csv
│   ├── ec2_cpu_utilization_77c1ca.csv
│   ├── ec2_cpu_utilization_825cc2.csv
│   ├── ec2_cpu_utilization_ac20cd.csv
│   ├── ec2_cpu_utilization_c6585a.csv
│   ├── ec2_cpu_utilization_fe7f93.csv
│   ├── ec2_disk_write_bytes_1ef3de.csv
│   ├── ec2_disk_write_bytes_c0d644.csv
│   ├── ec2_network_in_257a54.csv
│   ├── elb_request_count_8c0756.csv
│   ├── grok_asg_anomaly.csv
│   ├── iio_us-east-1_i-a2eb1cd9_NetworkIn.csv
│   ├── rds_cpu_utilization_cc0c53.csv
│   └── rds_cpu_utilization_e47b3b.csv
│
├── output/                       # Auto-created; stores plots + results
│   ├── 02_score_distribution.png
│   ├── 03_anomalies_per_source.png
│   ├── 04_feature_importance.png
│   ├── 05_cv_stability.png
│   └── anomaly_results.csv
│
└── README.md
```

---

## 🧠 What is Isolation Forest?

**Isolation Forest** is an unsupervised anomaly detection algorithm. It builds an ensemble of random decision trees that *isolate* observations by randomly selecting a feature and a split value. Anomalies are isolated in **fewer splits** because they are:
- Rare in occurrence
- Different in value from the majority of the data

Each point gets an **anomaly score** — the closer to `-1`, the more anomalous.

---

## 📊 Dataset Overview

| Metric Category | Files | Description |
|---|---|---|
| `cpu` | 8 files | EC2 CPU utilization (%) — 40,320 rows |
| `disk` | 2 files | EC2 disk write bytes — 8,762 rows |
| `network` | 2 files | EC2 & instance-level network-in bytes — 8,762 rows |
| `elb` | 1 file | ELB request count — 4,032 rows |
| `rds` | 2 files | RDS CPU utilization — included in `cpu` group |
| `other` | 2 files | ASG anomaly + IIO network — 5,864 rows |

- **Total rows:** 67,740
- **Date range:** 2013-10-09 to 2014-04-24
- **Missing values:** 0

---

## ⚙️ Pipeline — Step by Step

### Step 1 — Data Loading
- Reads all 17 CSV files using `glob`
- Parses timestamps, sorts chronologically
- Auto-tags each file with a `metric` category (`cpu`, `disk`, `network`, `elb`, `other`)

### Step 2 — Feature Engineering
Each time-series gets 16 features engineered per row:

| Feature | Description |
|---|---|
| `value` | Raw metric value |
| `rolling_mean/std/min/max/range` | Rolling window stats (window=12) |
| `z_score` | How many std-devs from rolling mean |
| `lag_1`, `lag_2`, `lag_3` | Previous 1/2/3 timestep values |
| `diff_1`, `diff_2` | First and second-order differences |
| `hour_sin/cos` | Cyclical encoding of hour of day |
| `dow_sin/cos` | Cyclical encoding of day of week |

### Step 3 — Preprocessing
- `StandardScaler` applied → all 16 features normalized to mean=0, std=1
- Feature matrix shape: **(67,740 × 16)**

### Step 4 — Hyperparameter Tuning with Cross-Validation
- **Random search** over 24–25 parameter combinations using `ParameterSampler`
- **Time-aware CV** (chronological folds — no data leakage)
- Scored on a **composite metric**: `mean_score / (1 + std_ratio)` — rewards confidence and penalizes instability

**Best parameters found:**
```python
{
  'n_estimators': 50,
  'max_samples': 256,
  'max_features': 0.8,
  'contamination': 0.01,
  'bootstrap': False
}
```

### Step 5 — Model Training & Prediction
- Final `IsolationForest` trained on full 67,740-row scaled dataset
- Outputs:
  - `anomaly_label`: `1` = normal, `-1` = anomaly
  - `anomaly_score`: lower = more anomalous
  - `is_anomaly`: binary flag (0/1)

**Results:**
- Total anomalies: **678 (1.00%)**
- Highest anomaly rate: `ec2_disk_write_bytes_c0d644` at **10.74%**

### Step 6 — Evaluation & Visualisation
Four plots are generated and saved to `./output/`:

| Plot | Description |
|---|---|
| `02_score_distribution.png` | Histogram + boxplot of anomaly scores for normal vs anomaly |
| `03_anomalies_per_source.png` | Time-series plot for all 17 sources with anomalies highlighted in red |
| `04_feature_importance.png` | Correlation of each feature with anomaly score (bar chart) |
| `05_cv_stability.png` | Anomaly rate stability across CV folds |

Final results exported to `output/anomaly_results.csv`.

---

## ⚙️ Requirements

```
Python >= 3.8
numpy
pandas
scikit-learn
matplotlib
jupyter
```

Install all at once:
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/anomaly-detection-isolation-tree.git
cd anomaly-detection-isolation-tree

# 2. Install dependencies
pip install numpy pandas scikit-learn matplotlib jupyter

# 3. Launch notebook
jupyter notebook isolation_tree.ipynb
```

Run all cells top to bottom. Output plots and CSV will appear in `./output/`.

---

## 📈 Key Results

| Source | Anomalies | Rate |
|---|---|---|
| ec2_disk_write_bytes_c0d644 | 433 | 10.74% |
| ec2_disk_write_bytes_1ef3de | 231 | 4.88% |
| iio_us-east-1_i-a2eb1cd9_NetworkIn | 12 | 0.97% |
| ec2_network_in_257a54 | 2 | 0.05% |
| All CPU / ELB / RDS sources | 0 | 0.00% |

---

## 📄 License

MIT License

---
