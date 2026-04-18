# Day 3 — Exploratory Data Analysis & Feature Engineering
# =========================================================
# Run this as a Jupyter notebook (recommended) or plain Python.
# To convert: jupyter nbconvert --to notebook --execute eda.py
#
# If using as .py, all plots will display sequentially.
# Recommended: copy into a .ipynb and run cell by cell.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# =============================================================================
# 1. LOAD & BASIC INSPECTION
# =============================================================================

df = pd.read_csv("ed_dataset.csv")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Simulations    : {df['sim_id'].nunique():,}")
print(f"  Breach rate    : {df['breached'].mean():.1%}")
print(f"  Missing values : {df.isnull().sum().sum()}")
print()
print(df.dtypes)
print()
print(df.describe().round(2).T)


# =============================================================================
# 2. TARGET VARIABLE DISTRIBUTION
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 2a. Breach count
counts = df["breached"].value_counts()
axes[0].bar(["No breach (0)", "Breach (1)"],
            [counts[0], counts[1]],
            color=["#4878CF", "#D65F5F"], edgecolor="white", width=0.5)
axes[0].set_title("Target variable distribution")
axes[0].set_ylabel("Patient count")
for i, v in enumerate([counts[0], counts[1]]):
    axes[0].text(i, v + 500, f"{v:,}\n({v/len(df):.1%})",
                 ha="center", fontsize=10)

# 2b. Total time distribution split by breach
df[df["breached"] == 0]["total_time_mins"].clip(upper=600).plot.hist(
    ax=axes[1], bins=60, alpha=0.6, color="#4878CF", label="No breach")
df[df["breached"] == 1]["total_time_mins"].clip(upper=1200).plot.hist(
    ax=axes[1], bins=60, alpha=0.6, color="#D65F5F", label="Breach")
axes[1].axvline(240, color="black", linestyle="--", linewidth=1.5, label="4-hour threshold")
axes[1].set_title("Total time distribution (clipped)")
axes[1].set_xlabel("Total time in ED (minutes)")
axes[1].legend()

plt.tight_layout()
plt.savefig("plot_01_target_distribution.png", bbox_inches="tight")
plt.show()
print("Saved: plot_01_target_distribution.png")


# =============================================================================
# 3. BREACH RATE BY KEY FEATURES
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a. Breach rate by number of doctors
breach_by_doc = df.groupby("num_doctors")["breached"].mean().reset_index()
axes[0, 0].bar(breach_by_doc["num_doctors"], breach_by_doc["breached"] * 100,
               color="#D65F5F", edgecolor="white")
axes[0, 0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0, 0].set_title("Breach rate by number of doctors")
axes[0, 0].set_xlabel("Number of doctors on shift")
axes[0, 0].set_ylabel("Breach rate (%)")

# 3b. Breach rate by triage level
breach_by_triage = df.groupby("triage_level")["breached"].mean().reset_index()
triage_labels = {1: "1\n(Critical)", 2: "2\n(Urgent)", 3: "3\n(Standard)",
                 4: "4\n(Minor)", 5: "5\n(Non-urgent)"}
axes[0, 1].bar(
    [triage_labels[t] for t in breach_by_triage["triage_level"]],
    breach_by_triage["breached"] * 100,
    color=["#D65F5F", "#E8855A", "#E8B55A", "#6CBF6C", "#4878CF"],
    edgecolor="white"
)
axes[0, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0, 1].set_title("Breach rate by triage level")
axes[0, 1].set_xlabel("Triage level")
axes[0, 1].set_ylabel("Breach rate (%)")

# 3c. Breach rate vs arrival rate (binned)
df["arrival_rate_bin"] = pd.cut(df["arrival_rate"], bins=6)
breach_by_arr = df.groupby("arrival_rate_bin", observed=True)["breached"].mean()
breach_by_arr.plot.bar(ax=axes[1, 0], color="#9B59B6", edgecolor="white")
axes[1, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[1, 0].set_title("Breach rate by arrival rate")
axes[1, 0].set_xlabel("Patients per hour")
axes[1, 0].set_ylabel("Breach rate")
axes[1, 0].tick_params(axis="x", rotation=30)

# 3d. Breach rate by queue length on arrival (binned)
df["queue_bin"] = pd.cut(df["queue_length_on_arrival"],
                         bins=[-1, 0, 2, 5, 10, 100],
                         labels=["0", "1–2", "3–5", "6–10", "10+"])
breach_by_q = df.groupby("queue_bin", observed=True)["breached"].mean()
breach_by_q.plot.bar(ax=axes[1, 1], color="#2ECC71", edgecolor="white")
axes[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[1, 1].set_title("Breach rate by queue length on arrival")
axes[1, 1].set_xlabel("Doctor queue length when patient arrived")
axes[1, 1].set_ylabel("Breach rate")
axes[1, 1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("plot_02_breach_rates_by_feature.png", bbox_inches="tight")
plt.show()
print("Saved: plot_02_breach_rates_by_feature.png")


# =============================================================================
# 4. CORRELATION HEATMAP
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

numeric_cols = ["num_doctors", "num_triage_nurses", "num_beds", "arrival_rate",
                "lab_capacity", "triage_level", "needs_lab", "arrival_hour",
                "queue_length_on_arrival", "wait_for_doctor_mins",
                "total_time_mins", "breached"]

corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask

sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("plot_03_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: plot_03_correlation_heatmap.png")

# Print top correlations with target
print("\nTop correlations with 'breached':")
target_corr = corr["breached"].drop("breached").sort_values(key=abs, ascending=False)
for feat, val in target_corr.items():
    bar = "█" * int(abs(val) * 20)
    direction = "+" if val > 0 else "-"
    print(f"  {feat:<30} {direction}{bar}  {val:+.3f}")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
# We derive new features that capture relationships the raw features miss.
# Each one has a clinical / operational justification.

print("\n" + "=" * 55)
print("FEATURE ENGINEERING")
print("=" * 55)

# ── 5a. Staff-to-demand ratio ─────────────────────────────────────────────────
# A hospital with 4 doctors and 20 arrivals/hr is very different from
# 4 doctors and 5 arrivals/hr. This single ratio captures that.
df["doctor_to_arrival_ratio"] = df["num_doctors"] / df["arrival_rate"]
print("✓ doctor_to_arrival_ratio  = num_doctors / arrival_rate")

# ── 5b. Bed pressure ─────────────────────────────────────────────────────────
# How saturated are the beds relative to expected occupancy?
# Higher = more bed pressure = longer waits
df["bed_pressure"] = df["arrival_rate"] / df["num_beds"]
print("✓ bed_pressure             = arrival_rate / num_beds")

# ── 5c. Is this a high-triage patient? ───────────────────────────────────────
# Triage levels 1 and 2 consume far more doctor time (see durations in sim).
# A binary flag is easier for tree models than a raw integer.
df["is_high_triage"] = (df["triage_level"] <= 2).astype(int)
print("✓ is_high_triage           = triage_level in {1, 2}")

# ── 5d. Is the ED congested on arrival? ──────────────────────────────────────
# Queue > 3 means a meaningful wait is almost guaranteed.
df["arrived_to_congestion"] = (df["queue_length_on_arrival"] >= 3).astype(int)
print("✓ arrived_to_congestion    = queue_length_on_arrival >= 3")

# ── 5e. Complexity score ─────────────────────────────────────────────────────
# Combines triage severity + lab requirement into one score.
# A level-1 patient needing a lab test is much harder to turn around quickly.
df["complexity_score"] = (6 - df["triage_level"]) + (df["needs_lab"] * 2)
print("✓ complexity_score         = (6 - triage_level) + (needs_lab × 2)")

# ── 5f. Shift pressure index ─────────────────────────────────────────────────
# Combines doctor shortage + high arrival rate into one stress index.
# Low ratio + high arrival = most stressful configuration.
df["shift_pressure"] = df["arrival_rate"] / (df["num_doctors"] * df["num_triage_nurses"])
print("✓ shift_pressure           = arrival_rate / (num_doctors × num_triage_nurses)")

print(f"\nNew feature columns added: 6")
print(f"Total features now: {df.shape[1]} columns")


# =============================================================================
# 6. VALIDATE NEW FEATURES
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# doctor_to_arrival_ratio vs breach rate
df["ratio_bin"] = pd.cut(df["doctor_to_arrival_ratio"], bins=8)
b1 = df.groupby("ratio_bin", observed=True)["breached"].mean()
b1.plot.bar(ax=axes[0], color="#3498DB", edgecolor="white")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[0].set_title("Breach rate vs doctor/arrival ratio")
axes[0].set_xlabel("Doctors per arrival/hr")
axes[0].tick_params(axis="x", rotation=35)

# complexity_score vs breach rate
b2 = df.groupby("complexity_score")["breached"].mean()
b2.plot.bar(ax=axes[1], color="#E74C3C", edgecolor="white")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[1].set_title("Breach rate vs complexity score")
axes[1].set_xlabel("Complexity score")

# shift_pressure vs breach rate
df["pressure_bin"] = pd.cut(df["shift_pressure"], bins=8)
b3 = df.groupby("pressure_bin", observed=True)["breached"].mean()
b3.plot.bar(ax=axes[2], color="#27AE60", edgecolor="white")
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
axes[2].set_title("Breach rate vs shift pressure index")
axes[2].set_xlabel("Shift pressure")
axes[2].tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.savefig("plot_04_engineered_features.png", bbox_inches="tight")
plt.show()
print("Saved: plot_04_engineered_features.png")


# =============================================================================
# 7. POINT OF INTEREST — wait_for_doctor_mins
# =============================================================================
# This feature is very predictive but we need to be careful:
# in a real deployment you wouldn't know wait_for_doctor_mins at
# prediction time (before the patient has seen the doctor).
#
# DECISION: We KEEP it for now as an analysis feature, but we will
# train two model versions:
#   - Version A: all features (post-hoc analysis)
#   - Version B: exclude wait_for_doctor_mins and total_time_mins (real-time prediction)
#
# Version B is the deployable one. Version A shows the upper bound.

print("\n" + "=" * 55)
print("LEAKAGE AUDIT")
print("=" * 55)
print("""
  wait_for_doctor_mins  — LEAKAGE RISK in real-time scenario.
    You don't know this until after the patient waited.
    → Excluded from Version B (real-time model).

  total_time_mins       — DIRECT LEAKAGE (it IS the target, just continuous).
    → Excluded from BOTH versions.

  queue_length_on_arrival — SAFE. Known at arrival time.
  arrival_hour            — SAFE. Known at arrival time.
  triage_level            — SAFE. Known after triage (minutes into visit).
  needs_lab               — SAFE. Known after doctor assessment.
                            Acceptable approximation for early prediction.
""")


# =============================================================================
# 8. SAVE FINAL FEATURE SET
# =============================================================================

# Drop temporary binning columns created for EDA
df.drop(columns=["arrival_rate_bin", "queue_bin", "ratio_bin", "pressure_bin"],
        inplace=True, errors="ignore")

# Full feature set (for analysis)
FEATURES_FULL = [
    "num_doctors", "num_triage_nurses", "num_beds", "arrival_rate", "lab_capacity",
    "triage_level", "needs_lab", "arrival_hour", "queue_length_on_arrival",
    "wait_for_doctor_mins",
    # engineered
    "doctor_to_arrival_ratio", "bed_pressure", "is_high_triage",
    "arrived_to_congestion", "complexity_score", "shift_pressure",
]

# Real-time feature set (deployable — no wait time known yet)
FEATURES_REALTIME = [
    "num_doctors", "num_triage_nurses", "num_beds", "arrival_rate", "lab_capacity",
    "triage_level", "needs_lab", "arrival_hour", "queue_length_on_arrival",
    # engineered
    "doctor_to_arrival_ratio", "bed_pressure", "is_high_triage",
    "arrived_to_congestion", "complexity_score", "shift_pressure",
]

TARGET = "breached"

df.to_csv("ed_dataset_engineered.csv", index=False)
print(f"Engineered dataset saved: ed_dataset_engineered.csv")
print(f"  Full feature set    : {len(FEATURES_FULL)} features")
print(f"  Real-time feat set  : {len(FEATURES_REALTIME)} features")
print(f"  Target              : {TARGET}")
print(f"  Total rows          : {len(df):,}")

# Save feature lists for Day 4 to import
import json
with open("feature_config.json", "w") as f:
    json.dump({
        "features_full": FEATURES_FULL,
        "features_realtime": FEATURES_REALTIME,
        "target": TARGET
    }, f, indent=2)
print(f"  Feature config saved: feature_config.json")
