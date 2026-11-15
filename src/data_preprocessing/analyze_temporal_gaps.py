
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD QUALITY-CHECKED DATA
# ==========================================

INPUT_FILE = Path(
    "data/processed/case_1_quality_checked.csv"
)

OUTPUT_FILE = Path(
    "data/processed/case_1_temporal_gap_report.csv"
)

df = pd.read_csv(INPUT_FILE)

SIGNALS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

# ==========================================
# 2. FUNCTION TO ANALYZE CONTIGUOUS PERIODS
# ==========================================

def get_run_lengths(mask):

    groups = mask.ne(mask.shift()).cumsum()

    return (
        mask.groupby(groups)
        .agg(["first", "size"])
    )


def analyze_signal(data, column):

    valid_mask = data[column].notna()

    valid_runs = get_run_lengths(valid_mask)
    missing_runs = get_run_lengths(~valid_mask)

    valid_lengths = valid_runs.loc[
        valid_runs["first"], "size"
    ]

    missing_lengths = missing_runs.loc[
        missing_runs["first"], "size"
    ]

    total = len(data)
    valid_count = int(valid_mask.sum())
    missing_count = total - valid_count

    return {
        "Signal": column,
        "Total_seconds": total,
        "Valid_seconds": valid_count,
        "Missing_seconds": missing_count,
        "Valid_percentage": round(
            valid_count / total * 100, 2
        ),
        "Missing_percentage": round(
            missing_count / total * 100, 2
        ),
        "Missing_periods": len(missing_lengths),
        "Longest_missing_gap_seconds": (
            int(missing_lengths.max())
            if len(missing_lengths) else 0
        ),
        "Average_missing_gap_seconds": round(
            missing_lengths.mean(), 2
        ) if len(missing_lengths) else 0,
        "Longest_valid_segment_seconds": (
            int(valid_lengths.max())
            if len(valid_lengths) else 0
        )
    }


# ==========================================
# 3. ANALYZE INDIVIDUAL SIGNALS
# ==========================================

results = []

print("\n========== TEMPORAL GAP ANALYSIS ==========")

for signal in SIGNALS:

    report = analyze_signal(df, signal)

    results.append(report)

    print(f"\n{signal}")

    for key, value in report.items():

        if key != "Signal":
            print(f"{key}: {value}")

# ==========================================
# 4. SIMULTANEOUS SIGNAL AVAILABILITY
# ==========================================

core = [
    "Solar8000/ART_MBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

all_five = SIGNALS

print("\n========== SIMULTANEOUS AVAILABILITY ==========")

for name, columns in [
    ("Core signals", core),
    ("All five signals", all_five)
]:

    complete_mask = df[columns].notna().all(axis=1)

    print(f"\n{name}")
    print("Valid seconds:", int(complete_mask.sum()))
    print(
        "Valid percentage:",
        round(complete_mask.mean() * 100, 2)
    )

    runs = get_run_lengths(complete_mask)

    valid_runs = runs.loc[
        runs["first"], "size"
    ]

    print(
        "Longest continuous segment:",
        int(valid_runs.max()) if len(valid_runs) else 0,
        "seconds"
    )

# ==========================================
# 5. SAVE REPORT
# ==========================================

report_df = pd.DataFrame(results)

report_df.to_csv(OUTPUT_FILE, index=False)

print("\nReport saved:", OUTPUT_FILE)

print("\nTemporal gap analysis completed.")
