
import os
import glob
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "data/processed/extracted_cases"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

# Broad physiological screening limits.
# These flag suspicious values; they do not automatically
# classify every flagged value as an artifact.
RANGES = {
    TRACKS[0]: (20, 200),
    TRACKS[1]: (40, 250),
    TRACKS[2]: (10, 150),
    TRACKS[3]: (20, 250),
    TRACKS[4]: (50, 100)
}

MIN_DURATION_MIN = 60
INTERVAL = 2

# ==========================================
# QUALITY CHECK
# ==========================================

files = glob.glob(
    os.path.join(INPUT_DIR, "case_*.csv")
)

results = []

print("Total extracted files:", len(files))

for index, file in enumerate(files, start=1):

    try:

        df = pd.read_csv(file)

        case_id = int(
            os.path.basename(file)
            .replace("case_", "")
            .replace(".csv", "")
        )

        required = ["time_seconds"] + TRACKS

        if not all(col in df.columns for col in required):
            raise ValueError("Missing required columns")

        duration_min = (
            df["time_seconds"].max()
            - df["time_seconds"].min()
        ) / 60

        result = {
            "caseid": case_id,
            "rows": len(df),
            "duration_min": duration_min
        }

        for track in TRACKS:

            values = pd.to_numeric(
                df[track],
                errors="coerce"
            )

            lower, upper = RANGES[track]

            valid = values.between(
                lower, upper
            )

            result[track + "_valid_pct"] = (
                valid.mean() * 100
            )

        # MAP below 65 is a candidate hypotension sample.
        # Sustained-event labeling is performed later.
        map_values = pd.to_numeric(
            df[TRACKS[0]],
            errors="coerce"
        )

        result["MAP_below_65_count"] = (
            (map_values < 65)
            & map_values.notna()
        ).sum()

        result["status"] = "SUCCESS"
        result["error"] = ""

        results.append(result)

    except Exception as e:

        results.append({
            "caseid": case_id if "case_id" in locals() else -1,
            "status": "FAILED",
            "error": str(e)
        })

    if index % 50 == 0:
        print(f"Checked {index}/{len(files)} files")

# ==========================================
# SAVE REPORT
# ==========================================

report = pd.DataFrame(results)

report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "patient_quality_report.csv"
    ),
    index=False
)

print("\nQUALITY CHECK COMPLETED")

print("Total files:", len(files))
print(
    "Successful:",
    (report["status"] == "SUCCESS").sum()
)
print(
    "Failed:",
    (report["status"] == "FAILED").sum()
)

print("\nDuration summary:")
print(
    report.loc[
        report["status"] == "SUCCESS",
        "duration_min"
    ].describe()
)

print("\nSaved: data/processed/patient_quality_report.csv")
