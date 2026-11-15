
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD QUALITY-CHECKED DATA
# ==========================================

INPUT_FILE = "data/processed/case_1_quality_checked.csv"

df = pd.read_csv(INPUT_FILE)

MAP = "Solar8000/ART_MBP"
SBP = "Solar8000/ART_SBP"
DBP = "Solar8000/ART_DBP"
HR = "Solar8000/HR"
SPO2 = "Solar8000/PLETH_SPO2"

# ==========================================
# 2. BROAD SCREENING RANGES
# These are investigation thresholds,
# not final clinical exclusion criteria.
# ==========================================

ranges = {
    MAP: (20, 200),
    SBP: (30, 300),
    DBP: (10, 200),
    HR: (20, 250),
    SPO2: (50, 100)
}

print("\n========== PHYSIOLOGICAL RANGE ANALYSIS ==========")

for col, (lower, upper) in ranges.items():

    invalid = (
        (df[col] < lower) |
        (df[col] > upper)
    )

    invalid_count = invalid.sum()

    print(f"\n{col}")
    print("Below lower bound:", (df[col] < lower).sum())
    print("Above upper bound:", (df[col] > upper).sum())
    print("Total suspicious:", invalid_count)

# ==========================================
# 3. BLOOD PRESSURE RELATIONSHIPS
# ==========================================

bp = df[[MAP, SBP, DBP]]

complete_bp = bp.dropna()

print("\n========== BLOOD PRESSURE CONSISTENCY ==========")

print("Complete BP observations:", len(complete_bp))

print(
    "SBP <= DBP:",
    (complete_bp[SBP] <= complete_bp[DBP]).sum()
)

print(
    "MAP outside DBP-SBP interval:",
    (
        (complete_bp[MAP] < complete_bp[DBP]) |
        (complete_bp[MAP] > complete_bp[SBP])
    ).sum()
)

# ==========================================
# 4. SIMULTANEOUS SIGNAL COVERAGE
# ==========================================

print("\n========== SIMULTANEOUS SIGNAL COVERAGE ==========")

core_signals = [MAP, HR, SPO2]

complete_core = df[core_signals].notna().all(axis=1)

print("All core signals available:", complete_core.sum())

print(
    "All core signals available (%):",
    round(complete_core.mean() * 100, 2)
)

all_signals = [MAP, SBP, DBP, HR, SPO2]

complete_all = df[all_signals].notna().all(axis=1)

print("All five signals available:", complete_all.sum())

# ==========================================
# 5. SAVE DIAGNOSTIC FLAGS
# ==========================================

diagnostic = df.copy()

for col, (lower, upper) in ranges.items():

    diagnostic[f"{col}_range_flag"] = (
        (df[col] < lower) |
        (df[col] > upper)
    )

diagnostic["BP_relationship_flag"] = (
    (
        (df[SBP] <= df[DBP]) |
        (df[MAP] < df[DBP]) |
        (df[MAP] > df[SBP])
    )
    & df[[MAP, SBP, DBP]].notna().all(axis=1)
)

OUTPUT_FILE = (
    "data/processed/case_1_physiological_diagnostics.csv"
)

diagnostic.to_csv(OUTPUT_FILE, index=False)

print("\nDiagnostic file saved:", OUTPUT_FILE)
print("\nPhysiological investigation completed.")
