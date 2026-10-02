#!/usr/bin/env python3
"""Normative modeling for posterior EEG measures in CLN3 disease.

Fits Bayesian linear-regression normative models to healthy-volunteer EEG
features and applies them to baseline CLN3 observations.
"""

import argparse
import copy
import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pcntoolkit import (
    BLR,
    NormativeModel,
    NormData,
    plot_centiles,
)
import pcntoolkit.util.output
import seaborn as sns
import numpy as np
from scipy.stats import norm

def parse_args():
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Fit EEG normative models and quantify CLN3 deviations."
    )
    parser.add_argument(
        "--hv-data",
        type=Path,
        default=Path("data/NovemberHVDataStaticAT.csv"),
        help="Healthy-volunteer feature table (default: data/NovemberHVDataStaticAT.csv).",
    )
    parser.add_argument(
        "--cln3-data",
        type=Path,
        default=Path("data/NovemberCLN3DataAT.csv"),
        help="CLN3 feature table (default: data/NovemberCLN3DataAT.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/CLN3_normative_quantification.csv"),
        help="Output CSV for subject-level normative results.",
    )
    return parser.parse_args()


ARGS = parse_args()
ARGS.output.parent.mkdir(parents=True, exist_ok=True)


# ---- Display / warnings config ----
sns.set_style("darkgrid", {"axes.grid": False})
plt.rcParams.update({"axes.grid": False, "grid.alpha": 0.0})
plt.close("all")

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None
pcntoolkit.util.output.Output.set_show_messages(False)


# ---------- Helpers ----------
def _remove_grids(fig=None):
    if fig is None:
        fig = plt.gcf()
    for ax in fig.get_axes():
        ax.grid(False)
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
    fig.canvas.draw_idle()
    return fig


def normalize_sex(val):
    s = str(val).strip().lower()
    if s in {"f", "female", "0", "0.0"}:
        return "F"
    if s in {"m", "male", "1", "1.0"}:
        return "M"
    return pd.NA


def harmonize_columns(df):
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .map(str)
        .str.strip()
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    alias_map = {
        "Subject": "sub_id",
        "Gender": "Sex",
        "Age": "Age_months",
        "Age (months)": "Age_months",
        "PeakFreq": "PeakFreq_Posterior",
        "AlphaThetaRatio": "Alpha_Theta_Ratio",
    }
    to_rename = {k: v for k, v in alias_map.items()
                 if k in df.columns and v not in df.columns}
    return df.rename(columns=to_rename)


RESPONSES = [
    "PeakFreq_Posterior",
    "Exponent",
    "Offset",
    "Alpha_Theta_Ratio",
    "Alpha_Theta_Ratio_log",
]

# =======================
# HV DATA
# =======================
df = pd.read_csv(ARGS.hv_data, header=2)
df = harmonize_columns(df)

required = [
    "Age_months",
    "PeakFreq_Posterior",
    "Exponent",
    "Offset",
    "Alpha_Theta_Ratio",
    "Sex",
    "sub_id",
]

for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")

for col in [
    "Age_months",
    "PeakFreq_Posterior",
    "Exponent",
    "Offset",
    "Alpha_Theta_Ratio",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Alpha_Theta_Ratio_log"] = np.log1p(df["Alpha_Theta_Ratio"])
df["sub_id"] = df["sub_id"].astype("string")
df["Sex"] = df["Sex"].map(normalize_sex).astype("string")

df.dropna(subset=required + ["Alpha_Theta_Ratio_log"], inplace=True)

norm_data = NormData.from_dataframe(
    "EEG_dataset",
    df,
    ["Age_months"],
    ["Sex"],
    RESPONSES,
    "sub_id",
)

train, hv_test = norm_data.train_test_split([0.8, 0.2])

# =======================
# MODEL FIT
# =======================
base_model = NormativeModel(BLR(), inscaler="standardize", outscaler="standardize")
base_model.fit(train)

# Create independent copies
model_HV = copy.deepcopy(base_model)
model_CLN3 = copy.deepcopy(base_model)

# Plotting ranges
model_HV.covariate_ranges["Age_months"]["min"] = 10
model_HV.covariate_ranges["Age_months"]["max"] = 210

model_CLN3.covariate_ranges["Age_months"]["min"] = 30
model_CLN3.covariate_ranges["Age_months"]["max"] = 270

# =======================
# HV CENTILES
# =======================
model_HV.predict(norm_data)

plot_centiles(
    model_HV,
    scatter_data=norm_data,
    covariate="Age_months",
    scatter_kwargs={
        "color": "blue",
        "marker": "o",
        "edgecolor": "blue",
        "linewidth": 0,
    },
)

plt.xlim(25, 270)

fig = plt.gcf()
for ax in fig.get_axes():
    ax.set_title("")  # remove automatic titles
    ax.set_xlabel("Age (months)")
    
    ylabel_map = {
        "PeakFreq_Posterior": "PDP",
        "Alpha_Theta_Ratio": "Alpha/Theta (ratio)",
        "Alpha_Theta_Ratio_log": "Alpha/Theta (log ratio)",
        "Exponent": "Exponent",
        "Offset": "Offset",
    }
    
    current_ylabel = ax.get_ylabel()
    if current_ylabel in ylabel_map:
        ax.set_ylabel(ylabel_map[current_ylabel])

_remove_grids()
plt.show()


# =======================
# CLN3 DATA
# =======================
df_cln3 = pd.read_csv(ARGS.cln3_data, header=2)
df_cln3 = harmonize_columns(df_cln3)

for col in [
    "Age_months",
    "PeakFreq_Posterior",
    "Exponent",
    "Offset",
    "Alpha_Theta_Ratio",
]:
    df_cln3[col] = pd.to_numeric(df_cln3[col], errors="coerce")

df_cln3["Alpha_Theta_Ratio_log"] = np.log1p(df_cln3["Alpha_Theta_Ratio"])
df_cln3["sub_id"] = df_cln3["sub_id"].astype("string")
df_cln3["Sex"] = df_cln3["Sex"].map(normalize_sex).astype("string")

df_cln3.dropna(subset=required + ["Alpha_Theta_Ratio_log"], inplace=True)

df_cln3["EEG"] = pd.to_numeric(df_cln3["EEG"], errors="coerce")
df_cln3_bl = df_cln3.query("EEG == 0").copy()

norm_data_cln3_bl = NormData.from_dataframe(
    "EEG_CLN3_baseline",
    df_cln3_bl,
    ["Age_months"],
    ["Sex"],
    RESPONSES,
    "sub_id",
)

model_CLN3.predict(norm_data_cln3_bl)

# =======================
# CLN3 CENTILES
# =======================
plot_centiles(
    model_CLN3,
    scatter_data=norm_data_cln3_bl,
    covariate="Age_months",
    scatter_kwargs={
        "color": "red",
        "marker": "^",
        "edgecolor": "red",
        "linewidth": 0,
    },
)

plt.xlim(5, 250)

fig = plt.gcf()
for ax in fig.get_axes():
    ax.set_title("")  # remove automatic titles
    ax.set_xlabel("Age (months)")
    
    ylabel_map = {
        "PeakFreq_Posterior": "PDP",
        "Alpha_Theta_Ratio": "Alpha/Theta (ratio)",
        "Alpha_Theta_Ratio_log": "Alpha/Theta (log ratio)",
        "Exponent": "Exponent",
        "Offset": "Offset",
    }
    
    current_ylabel = ax.get_ylabel()
    if current_ylabel in ylabel_map:
        ax.set_ylabel(ylabel_map[current_ylabel])

_remove_grids()
plt.show()


# =======================
# PREDICTIONS + NUMBERS
# =======================

pred_data = model_CLN3.predict(norm_data_cln3_bl)

# Extract prediction arrays
Yhat = pred_data["Yhat"].values        # predicted mean
Z = pred_data["Z"].values              # deviation scores
centiles_full = pred_data["centiles"].values  # shape: (centile, obs, response)

# Derive subject-specific centile position from the deviation score.

centiles = norm.cdf(Z) * 100

# Build subject-level results table.
results = pd.DataFrame()

for i, resp in enumerate(RESPONSES):
    results[f"{resp}_observed"] = df_cln3_bl[resp].values
    results[f"{resp}_predicted"] = Yhat[:, i]
    results[f"{resp}_Z"] = Z[:, i]
    results[f"{resp}_centile"] = centiles[:, i]

results["sub_id"] = df_cln3_bl["sub_id"].values
results["Age_months"] = df_cln3_bl["Age_months"].values
results["Sex"] = df_cln3_bl["Sex"].values

results.to_csv(ARGS.output, index=False)

print("Normative quantification saved.")


# =======================
# PRINT SUBJECT-LEVEL RESULTS
# =======================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 3)

print("\n=== CLN3 Normative Quantification (Subject-Level) ===\n")
print(results)


# =======================
# GROUP-LEVEL NORMATIVE DEVIATION ANALYSIS
# =======================

from scipy.stats import ttest_1samp

print("\n=== Group-Level Summary: Normative Z-scores ===\n")

z_cols = [col for col in results.columns if col.endswith("_Z")]

group_summary = results[z_cols].agg(["mean", "std", "min", "max"])
print(group_summary)


print("\n=== One-sample t-tests against normative expectation (H0: mean Z = 0) ===\n")

test_rows = []

for col in z_cols:
    z = results[col].dropna()
    t, p = ttest_1samp(z, 0)

    test_rows.append({
        "Measure": col.replace("_Z", ""),
        "N": len(z),
        "Mean_Z": z.mean(),
        "SD_Z": z.std(),
        "Mean_abs_Z": z.abs().mean(),
        "t": t,
        "df": len(z) - 1,
        "p": p,
        "Min_Z": z.min(),
        "Max_Z": z.max(),
    })

test_summary = pd.DataFrame(test_rows)
print(test_summary)


print("\n=== Individual-Level Abnormality Burden ===\n")

abnormality_rows = []

for col in z_cols:
    z = results[col].dropna()
    n = len(z)

    n_outside_95 = (z.abs() > 1.96).sum()
    n_outside_99 = (z.abs() > 2.58).sum()

    abnormality_rows.append({
        "Measure": col.replace("_Z", ""),
        "N": n,
        "N_outside_95": n_outside_95,
        "Pct_outside_95": 100 * n_outside_95 / n,
        "N_outside_99": n_outside_99,
        "Pct_outside_99": 100 * n_outside_99 / n,
    })

abnormality_summary = pd.DataFrame(abnormality_rows)
print(abnormality_summary)
