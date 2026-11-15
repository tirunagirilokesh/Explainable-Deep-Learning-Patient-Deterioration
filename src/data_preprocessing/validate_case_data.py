
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================
# 1. LOAD ORIGINAL EXTRACTED DATA
# =====================================

INPUT_FILE = Path("data/raw/case_1_arterial_raw.csv")
OUTPUT_FILE = Path("data/processed/case_1_quality_checked.csv")

df = pd.read_csv(INPUT_FILE)

signals = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

# =====================================
# 2. COUNT INVALID VALUES
# =====================================

print("\n========== INVALID VALUE ANALYSIS ==========")

for col in signals:
    negative_count = (df[col] < 0).sum()
    zero_count = (df[col] == 0).sum()
    nan_count = df[col].isna().sum()

    print(f"\n{col}")
    print("Negative values:", negative_count)
    print("Zero values:", zero_count)
    print("NaN values:", nan_count)

# =====================================
# 3. CLEAN INVALID NON-POSITIVE VALUES
# =====================================

clean_df = df.copy()

for col in signals:
    clean_df[col] = pd.to_numeric(
        clean_df[col], errors="coerce"
    )

    clean_df.loc[clean_df[col] <= 0, col] = np.nan

# =====================================
# 4. MISSING VALUE REPORT
# =====================================

print("\n========== CLEANED DATA QUALITY ==========")

print("\nMissing values:")
print(clean_df[signals].isna().sum())

print("\nMissing percentages:")
print(
    (clean_df[signals].isna().mean() * 100).round(2)
)

print("\nValid statistical summary:")
print(clean_df[signals].describe())

# =====================================
# 5. SAVE DIAGNOSTIC DATA
# =====================================

OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)

clean_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("\nQuality validation completed.")
