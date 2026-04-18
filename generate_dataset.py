"""
Dataset Generation — 5,000 Simulation Runs
============================================
Imports the ED simulation and runs it 5,000 times with
different parameter combinations.

Each run = one ED configuration (staffing, arrival rate, etc.)
Each patient within a run = one row in the dataset.

Expected output: ~50,000–150,000 patient records
Target breach rate: 25–45%

Run this file once. It saves the dataset to ed_dataset.csv.
Takes roughly 3–8 minutes depending on your machine.
"""

import pandas as pd
import numpy as np
import time
import os
from ed_simulation import run_single_simulation, sample_params


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

NUM_SIMULATIONS = 5000
OUTPUT_FILE     = "ed_dataset.csv"
CHECKPOINT_EVERY = 500      # save progress every N runs in case of interruption
RANDOM_SEED_BASE = 42       # for reproducibility


# ─────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────

def generate_dataset(num_sims: int = NUM_SIMULATIONS) -> pd.DataFrame:

    all_records = []
    breach_counts = []
    skipped = 0
    start_time = time.time()

    print(f"Starting {num_sims:,} simulations...")
    print(f"Saving to: {OUTPUT_FILE}")
    print(f"Checkpoint every {CHECKPOINT_EVERY} runs\n")
    print(f"{'Sim':>6} | {'Doctors':>7} | {'Arr/hr':>6} | {'Patients':>8} | {'Breach%':>7} | {'Elapsed':>8} | {'ETA':>8}")
    print("─" * 72)

    for i in range(num_sims):
        seed = RANDOM_SEED_BASE + i
        params = sample_params(seed=seed)

        try:
            df = run_single_simulation(params)

            if len(df) == 0:
                skipped += 1
                continue

            all_records.append(df)
            breach_counts.append(df["breached"].mean())

            # Progress logging every 100 runs
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed          # sims per second
                eta = (num_sims - i - 1) / rate   # seconds remaining

                elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60):02d}s"
                eta_str     = f"{int(eta//60)}m{int(eta%60):02d}s"

                recent_breach = np.mean(breach_counts[-100:])
                total_records = sum(len(d) for d in all_records)

                print(f"{i+1:>6,} | {params['num_doctors']:>7} | "
                      f"{params['arrival_rate']:>6.1f} | "
                      f"{total_records:>8,} | "
                      f"{recent_breach:>6.1%} | "
                      f"{elapsed_str:>8} | "
                      f"{eta_str:>8}")

            # Checkpoint save
            if (i + 1) % CHECKPOINT_EVERY == 0:
                checkpoint = pd.concat(all_records, ignore_index=True)
                checkpoint.to_csv(OUTPUT_FILE, index=False)
                print(f"  ✓ Checkpoint saved — {len(checkpoint):,} records so far\n")

        except Exception as e:
            skipped += 1
            if skipped <= 5:  # only print first few errors
                print(f"  ✗ Sim {i+1} failed: {e}")

    # ── FINAL DATASET ──────────────────────────────────────────
    print("\nCombining all records...")
    dataset = pd.concat(all_records, ignore_index=True)

    # Add a simulation_id so we can trace any record back to its run
    # (useful for debugging and for groupby analysis later)
    sim_ids = []
    for idx, df in enumerate(all_records):
        sim_ids.extend([idx] * len(df))
    dataset.insert(0, "sim_id", sim_ids)

    return dataset


# ─────────────────────────────────────────────
# POST-GENERATION SUMMARY
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    total = len(df)
    breached = df["breached"].sum()
    breach_rate = df["breached"].mean()

    print("\n" + "═" * 50)
    print("DATASET GENERATION COMPLETE")
    print("═" * 50)
    print(f"  Total patient records : {total:,}")
    print(f"  Breach (1)            : {breached:,}  ({breach_rate:.1%})")
    print(f"  No breach (0)         : {total - breached:,}  ({1-breach_rate:.1%})")
    print(f"  Unique simulations    : {df['sim_id'].nunique():,}")
    print(f"  Features              : {df.shape[1]} columns")
    print()

    print("Feature ranges:")
    for col in ["num_doctors", "arrival_rate", "triage_level",
                "queue_length_on_arrival", "wait_for_doctor_mins", "total_time_mins"]:
        print(f"  {col:<30} min={df[col].min():.1f}  max={df[col].max():.1f}  "
              f"mean={df[col].mean():.1f}")

    print()
    breach_rate_val = df["breached"].mean()
    if breach_rate_val < 0.20:
        print("⚠  Breach rate below 20% — consider adjusting stress config weights")
    elif breach_rate_val > 0.55:
        print("⚠  Breach rate above 55% — dataset may be too skewed toward stress configs")
    else:
        print(f"✓  Breach rate {breach_rate_val:.1%} is in the healthy 20–55% range for ML training")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # If dataset already exists, ask before overwriting
    if os.path.exists(OUTPUT_FILE):
        ans = input(f"\n'{OUTPUT_FILE}' already exists. Overwrite? (y/n): ").strip().lower()
        if ans != "y":
            print("Aborted.")
            exit()

    dataset = generate_dataset()

    # Save final version
    dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal dataset saved to: {OUTPUT_FILE}")

    print_summary(dataset)

    # Quick peek at the data
    print("\nFirst 5 rows:")
    print(dataset.head().to_string(index=False))
