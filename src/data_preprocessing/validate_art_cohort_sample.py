
import vitaldb
import pandas as pd
import numpy as np

from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

TRACKS_FILE = "data/raw/tracks.csv"

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

SAMPLE_SIZE = 10
RANDOM_STATE = 42
TOLERANCE_SECONDS = 1.0

OUTPUT_FILE = (
    "data/processed/art_cohort_sample_validation.csv"
)

# =====================================================
# STEP 1: IDENTIFY ELIGIBLE ART CASES
# =====================================================

print("\n========== ART COHORT SAMPLE VALIDATION ==========")

tracks_df = pd.read_csv(TRACKS_FILE)

art_tracks = tracks_df[
    tracks_df["tname"].isin(TRACKS)
]

availability = (
    art_tracks.groupby("caseid")["tname"]
    .nunique()
)

eligible_cases = availability[
    availability == len(TRACKS)
].index.tolist()

print("Total metadata-eligible ART cases:", len(eligible_cases))

if len(eligible_cases) < SAMPLE_SIZE:
    raise ValueError("Not enough eligible cases for sample validation.")

# Include Case 1, which we have already inspected.
# Select the remaining cases reproducibly.
remaining_cases = [
    case_id for case_id in eligible_cases
    if case_id != 1
]

sample_cases = [1] + pd.Series(remaining_cases).sample(
    n=SAMPLE_SIZE - 1,
    random_state=RANDOM_STATE
).tolist()

print("\nSelected cases:", sample_cases)

# =====================================================
# STEP 2: INSPECT EACH CASE
# =====================================================

results = []

for index, case_id in enumerate(sample_cases, start=1):

    print(f"\n[{index}/{SAMPLE_SIZE}] Processing Case {case_id}")

    row = {
        "caseid": case_id,
        "status": "FAILED"
    }

    try:
        vf = vitaldb.VitalFile(case_id, TRACKS)

        track_frames = {}

        for track_name in TRACKS:

            track = vf.find_track(track_name)

            if track is None:
                raise ValueError(
                    f"Missing track: {track_name}"
                )

            records = track.recs

            timestamps = np.array(
                [record["dt"] for record in records],
                dtype=float
            )

            values = np.array(
                [record["val"] for record in records],
                dtype=float
            )

            # Preliminary sentinel screening only.
            valid_mask = (
                np.isfinite(timestamps) &
                np.isfinite(values) &
                (values > 0)
            )

            valid_df = pd.DataFrame({
                "timestamp": timestamps[valid_mask],
                "value": values[valid_mask]
            })

            valid_df = (
                valid_df.sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )

            track_frames[track_name] = valid_df

            short_name = track_name.split("/")[-1]

            row[f"{short_name}_records"] = len(records)
            row[f"{short_name}_valid"] = len(valid_df)

            row[f"{short_name}_valid_pct"] = round(
                100 * len(valid_df) / len(records), 2
            ) if len(records) else 0

            intervals = np.diff(
                valid_df["timestamp"].to_numpy()
            )

            row[f"{short_name}_median_interval"] = (
                round(float(np.median(intervals)), 3)
                if len(intervals) else np.nan
            )

            row[f"{short_name}_max_gap"] = (
                round(float(np.max(intervals)), 3)
                if len(intervals) else np.nan
            )

            row[f"{short_name}_gaps_over_10s"] = (
                int(np.sum(intervals > 10))
                if len(intervals) else 0
            )

        # =================================================
        # STEP 3: USE MAP AS THE REFERENCE TIMELINE
        # =================================================

        map_name = "Solar8000/ART_MBP"

        reference = track_frames[map_name].copy()

        reference = reference.rename(
            columns={"value": "MAP"}
        )

        row["valid_MAP_observations"] = len(reference)

        if len(reference) > 0:
            row["recording_duration_hours"] = round(
                (
                    reference["timestamp"].max()
                    - reference["timestamp"].min()
                ) / 3600,
                3
            )
        else:
            row["recording_duration_hours"] = 0

        complete_mask = pd.Series(
            True,
            index=reference.index
        )

        repeated_total = 0

        # =================================================
        # STEP 4: ALIGN THE FOUR SUPPORTING SIGNALS
        # =================================================

        for track_name in TRACKS:

            if track_name == map_name:
                continue

            short_name = track_name.split("/")[-1]

            source = track_frames[track_name].rename(
                columns={
                    "timestamp": "source_timestamp",
                    "value": short_name
                }
            )

            matched = pd.merge_asof(
                reference[["timestamp"]].sort_values(
                    "timestamp"
                ),
                source.sort_values("source_timestamp"),
                left_on="timestamp",
                right_on="source_timestamp",
                direction="nearest",
                tolerance=TOLERANCE_SECONDS
            )

            matched_values = matched[short_name].notna()

            complete_mask &= matched_values

            matched_timestamps = matched.loc[
                matched_values,
                "source_timestamp"
            ]

            repeated_count = int(
                matched_timestamps.duplicated().sum()
            )

            repeated_total += repeated_count

            row[f"{short_name}_match_pct"] = round(
                100 * matched_values.sum() / len(reference),
                2
            ) if len(reference) else 0

            row[f"{short_name}_repeated_matches"] = (
                repeated_count
            )

        row["complete_five_signal_observations"] = int(
            complete_mask.sum()
        )

        row["complete_five_signal_pct"] = round(
            100 * complete_mask.mean(),
            2
        ) if len(reference) else 0

        row["total_repeated_matches"] = repeated_total

        row["status"] = "SUCCESS"

        print(
            "Duration:",
            row["recording_duration_hours"],
            "hours"
        )

        print(
            "Complete five-signal coverage:",
            row["complete_five_signal_pct"],
            "%"
        )

    except Exception as error:

        row["error"] = str(error)

        print("ERROR:", error)

    results.append(row)

# =====================================================
# STEP 5: SAVE REPORT
# =====================================================

report = pd.DataFrame(results)

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

report.to_csv(OUTPUT_FILE, index=False)

print("\n========== SAMPLE VALIDATION SUMMARY ==========")

print(
    report[
        [
            "caseid",
            "status",
            "recording_duration_hours",
            "valid_MAP_observations",
            "complete_five_signal_pct",
            "total_repeated_matches"
        ]
    ].to_string(index=False)
)

print("\nSuccessful cases:", (report["status"] == "SUCCESS").sum())
print("Failed cases:", (report["status"] != "SUCCESS").sum())

print("\nReport saved to:", OUTPUT_FILE)
print("\nART sample validation completed.")
