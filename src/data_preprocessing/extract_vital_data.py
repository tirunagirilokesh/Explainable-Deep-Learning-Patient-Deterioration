
import os
import time
import numpy as np
import pandas as pd
import vitaldb
import traceback

# =====================================================
# 1. CONFIGURATION
# =====================================================

TRACKS_FILE = "data/raw/tracks.csv"
OUTPUT_DIR = "data/processed/extracted_cases"
VITAL_DIR = "data/vital_files"
REPORT_DIR = "reports"

MANIFEST_PATH = os.path.join(
    REPORT_DIR, "extraction_manifest.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VITAL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

INTERVAL = 2  # seconds

SAVE_VITAL_FILES = False

# None = all eligible cases
MAX_CASES = None

# Number of attempts per case
MAX_RETRIES = 2

# Small delay between attempts
RETRY_DELAY = 2

# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def get_output_path(case_id):
    return os.path.join(
        OUTPUT_DIR,
        f"case_{case_id}.csv"
    )


def validate_existing_csv(path):
    """
    Check whether an existing extracted CSV is usable.
    """

    try:
        if not os.path.exists(path):
            return False

        if os.path.getsize(path) == 0:
            return False

        header = pd.read_csv(path, nrows=0)

        required = [
            "time_seconds",
            "caseid"
        ] + TRACKS

        if not all(col in header.columns for col in required):
            return False

        # Confirm that at least one data row exists.
        sample = pd.read_csv(path, nrows=2)

        return len(sample) > 0

    except Exception:
        return False


def calculate_metrics(df):
    missing_pct = df[TRACKS].isna().mean() * 100

    return {
        "rows": len(df),
        "duration_hours": round(
            len(df) * INTERVAL / 3600, 3
        ),
        "MAP_missing_pct": missing_pct[TRACKS[0]],
        "SBP_missing_pct": missing_pct[TRACKS[1]],
        "DBP_missing_pct": missing_pct[TRACKS[2]],
        "HR_missing_pct": missing_pct[TRACKS[3]],
        "SpO2_missing_pct": missing_pct[TRACKS[4]]
    }


def save_manifest(results):
    """
    Save all accumulated results after every case.
    """

    report = pd.DataFrame(results.values())

    if not report.empty:
        report = report.sort_values("caseid")

    report.to_csv(MANIFEST_PATH, index=False)


# =====================================================
# 3. LOAD EXISTING MANIFEST FOR RESUMING
# =====================================================

print("\nLoading previous extraction progress...")

results = {}

if os.path.exists(MANIFEST_PATH):

    try:
        old_report = pd.read_csv(MANIFEST_PATH)

        for _, row in old_report.iterrows():
            results[int(row["caseid"])] = row.to_dict()

        print(
            "Previous manifest records:",
            len(results)
        )

    except Exception as e:
        print("Could not read previous manifest:", e)
        print("Starting with an empty manifest.")

else:
    print("No previous manifest found. Starting fresh.")


# =====================================================
# 4. SELECT ELIGIBLE ART CASES
# =====================================================

print("\nLoading VitalDB track metadata...")

tracks_df = pd.read_csv(TRACKS_FILE)

required_columns = {"caseid", "tname"}

if not required_columns.issubset(tracks_df.columns):
    raise ValueError(
        f"tracks.csv must contain {required_columns}"
    )

track_sets = {}

for track in TRACKS:

    track_sets[track] = set(
        tracks_df.loc[
            tracks_df["tname"] == track,
            "caseid"
        ].dropna().astype(int)
    )

eligible_cases = set.intersection(
    *track_sets.values()
)

case_ids = sorted(eligible_cases)

if MAX_CASES is not None:
    case_ids = case_ids[:MAX_CASES]

print("\nMetadata-eligible ART cases:", len(eligible_cases))
print("Cases selected for extraction:", len(case_ids))

# =====================================================
# 5. EXTRACTION LOOP
# =====================================================

successful = 0
failed = 0
skipped = 0

for index, case_id in enumerate(case_ids, start=1):

    print("\n" + "=" * 65)
    print(
        f"Processing case {case_id} "
        f"({index}/{len(case_ids)})"
    )
    print("=" * 65)

    output_path = get_output_path(case_id)

    # -------------------------------------------------
    # A. SKIP ALREADY VALID SUCCESSFUL CASES
    # -------------------------------------------------

    previous_status = str(
        results.get(case_id, {}).get("status", "")
    )

    if (
        previous_status == "SUCCESS"
        and validate_existing_csv(output_path)
    ):

        print("Already extracted and validated. Skipping.")

        skipped += 1
        continue

    # -------------------------------------------------
    # B. CHECK EXISTING CSV WITHOUT SUCCESS MANIFEST
    # -------------------------------------------------

    if (
        previous_status != "SUCCESS"
        and validate_existing_csv(output_path)
    ):

        print("Valid CSV already exists. Reusing it.")

        existing_df = pd.read_csv(output_path)

        metrics = calculate_metrics(existing_df)

        results[case_id] = {
            "caseid": case_id,
            "status": "SUCCESS",
            **metrics,
            "error": ""
        }

        skipped += 1
        save_manifest(results)
        continue

    # -------------------------------------------------
    # C. DOWNLOAD AND EXTRACT WITH RETRIES
    # -------------------------------------------------

    case_success = False
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"Download attempt {attempt}/{MAX_RETRIES}"
            )

            vf = vitaldb.VitalFile(
                case_id,
                TRACKS
            )

            # Verify required tracks
            available_tracks = vf.get_track_names()

            missing = [
                track for track in TRACKS
                if track not in available_tracks
            ]

            if missing:
                raise ValueError(
                    f"Missing tracks: {missing}"
                )

            # Resample onto a regular 2-second grid
            values = vf.to_numpy(
                TRACKS,
                interval=INTERVAL
            )

            if values is None or values.size == 0:
                raise ValueError("No data returned")

            if (
                values.ndim != 2
                or values.shape[1] != len(TRACKS)
            ):
                raise ValueError(
                    f"Unexpected array shape: {values.shape}"
                )

            # Build dataframe
            df = pd.DataFrame(
                values,
                columns=TRACKS
            )

            df.insert(
                0,
                "time_seconds",
                np.arange(len(df)) * INTERVAL
            )

            df.insert(
                1,
                "caseid",
                case_id
            )

            if df.empty:
                raise ValueError("Extracted dataframe is empty")

            # Save to temporary path first
            temp_path = output_path + ".tmp"

            df.to_csv(
                temp_path,
                index=False
            )

            # Replace final file only after successful write
            os.replace(
                temp_path,
                output_path
            )

            # Optionally save original VitalDB file
            if SAVE_VITAL_FILES:

                vital_path = os.path.join(
                    VITAL_DIR,
                    f"case_{case_id}.vital"
                )

                vf.to_vital(vital_path)

            # Calculate missing percentages
            metrics = calculate_metrics(df)

            results[case_id] = {
                "caseid": case_id,
                "status": "SUCCESS",
                **metrics,
                "error": ""
            }

            case_success = True

            print("\nSUCCESS")
            print("Rows:", len(df))
            print(
                "Duration (hours):",
                metrics["duration_hours"]
            )

            print(
                "MAP missing:",
                round(metrics["MAP_missing_pct"], 2),
                "%"
            )

            break

        except Exception as e:

            last_error = str(e)

            print(
                f"Attempt {attempt} failed:",
                last_error
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    # -------------------------------------------------
    # D. RECORD FINAL CASE STATUS
    # -------------------------------------------------

    if case_success:
        successful += 1

    else:

        failed += 1

        results[case_id] = {
            "caseid": case_id,
            "status": "FAILED",
            "rows": 0,
            "duration_hours": 0,
            "MAP_missing_pct": np.nan,
            "SBP_missing_pct": np.nan,
            "DBP_missing_pct": np.nan,
            "HR_missing_pct": np.nan,
            "SpO2_missing_pct": np.nan,
            "error": last_error
        }

        print("CASE FAILED AFTER ALL RETRIES")

    # Save progress after every processed case
    save_manifest(results)

    print(
        f"\nRun progress: {index}/{len(case_ids)}"
    )
    print(
        f"New successes: {successful} | "
        f"New failures: {failed} | "
        f"Skipped: {skipped}"
    )


# =====================================================
# 6. FINAL REPORT
# =====================================================

report = pd.DataFrame(results.values())

if not report.empty:
    report = report.sort_values("caseid")

    # Save final manifest
    report.to_csv(
        MANIFEST_PATH,
        index=False
    )

print("\n" + "=" * 65)
print("EXTRACTION SUMMARY")
print("=" * 65)

print("Metadata-eligible cases:", len(eligible_cases))
print("Cases selected:", len(case_ids))
print("New successes:", successful)
print("New failures:", failed)
print("Skipped existing cases:", skipped)

if not report.empty:

    print(
        "Total successful in manifest:",
        (report["status"] == "SUCCESS").sum()
    )

    print(
        "Total failed in manifest:",
        (report["status"] == "FAILED").sum()
    )

    failed_cases = report[
        report["status"] == "FAILED"
    ]

    if not failed_cases.empty:

        print("\nFailure details:")

        print(
            failed_cases[
                ["caseid", "error"]
            ].to_string(index=False)
        )

print("\nManifest saved:")
print(MANIFEST_PATH)

print("\nExtracted CSV directory:")
print(OUTPUT_DIR)

print("\nExtraction process finished.")
