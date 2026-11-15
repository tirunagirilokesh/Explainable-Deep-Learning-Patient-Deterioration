
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

SAMPLE_SIZE = 30
RANDOM_STATE = 42
TOLERANCE_SECONDS = 1.0
CONTINUITY_GAP_SECONDS = 3.0

OUTPUT_FILE = (
    "data/processed/art_cohort_full_validation.csv"
)


# =====================================================
# ONE-TO-ONE TIMESTAMP MATCHING
# =====================================================

def match_one_to_one(reference_times, source_times, tolerance):

    reference_times = np.asarray(reference_times, dtype=float)
    source_times = np.asarray(source_times, dtype=float)

    matched_indices = np.full(
        len(reference_times), -1, dtype=int
    )

    differences = np.full(
        len(reference_times), np.nan, dtype=float
    )

    j = 0

    for i, ref_time in enumerate(reference_times):

        # Discard source timestamps that are too early.
        while (
            j < len(source_times)
            and source_times[j] < ref_time - tolerance
        ):
            j += 1

        if j >= len(source_times):
            break

        # Compare current and next source observation.
        candidates = []

        if abs(source_times[j] - ref_time) <= tolerance:
            candidates.append(j)

        if j + 1 < len(source_times):
            if abs(source_times[j + 1] - ref_time) <= tolerance:
                candidates.append(j + 1)

        if not candidates:
            continue

        best_index = min(
            candidates,
            key=lambda k: abs(source_times[k] - ref_time)
        )

        matched_indices[i] = best_index

        differences[i] = abs(
            source_times[best_index] - ref_time
        )

        # Advance beyond the selected source record.
        j = best_index + 1

    return matched_indices, differences


# =====================================================
# CONTINUOUS SEGMENT ANALYSIS
# =====================================================

def get_continuous_segment_stats(timestamps, gap_threshold):

    timestamps = np.asarray(timestamps, dtype=float)

    if len(timestamps) == 0:
        return 0, 0.0

    timestamps = np.sort(timestamps)

    if len(timestamps) == 1:
        return 1, 0.0

    gaps = np.diff(timestamps)

    split_positions = np.where(
        gaps > gap_threshold
    )[0] + 1

    segments = np.split(timestamps, split_positions)

    longest_seconds = max(
        segment[-1] - segment[0]
        for segment in segments
        if len(segment) > 0
    )

    return len(segments), float(longest_seconds)


# =====================================================
# STEP 1: SELECT ART CASES
# =====================================================

print("\n========== ART COHORT VALIDATION V2 ==========")

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

print("Metadata-eligible ART cases:", len(eligible_cases))

remaining_cases = [
    case_id for case_id in eligible_cases
    if case_id != 1
]

sample_cases = [1] + pd.Series(remaining_cases).sample(
    n=SAMPLE_SIZE - 1,
    random_state=RANDOM_STATE
).tolist()

print("Selected cases:", sample_cases)


# =====================================================
# STEP 2: PROCESS CASES
# =====================================================

results = []

for case_number, case_id in enumerate(sample_cases, start=1):

    print(
        f"\n[{case_number}/{SAMPLE_SIZE}] "
        f"Processing Case {case_id}"
    )

    row = {
        "caseid": case_id,
        "status": "FAILED"
    }

    try:

        vf = vitaldb.VitalFile(case_id, TRACKS)

        track_data = {}
        all_original_timestamps = []

        # ---------------------------------------------
        # Extract original records
        # ---------------------------------------------

        for track_name in TRACKS:

            track = vf.find_track(track_name)

            if track is None:
                raise ValueError(
                    f"Missing track: {track_name}"
                )

            records = track.recs

            timestamps = np.array(
                [r["dt"] for r in records],
                dtype=float
            )

            values = np.array(
                [r["val"] for r in records],
                dtype=float
            )

            finite_timestamps = timestamps[
                np.isfinite(timestamps)
            ]

            if len(finite_timestamps):
                all_original_timestamps.extend(
                    finite_timestamps.tolist()
                )

            valid_mask = (
                np.isfinite(timestamps)
                & np.isfinite(values)
                & (values > 0)
            )

            valid_data = pd.DataFrame({
                "timestamp": timestamps[valid_mask],
                "value": values[valid_mask]
            })

            valid_data = (
                valid_data.sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )

            track_data[track_name] = valid_data

            short_name = track_name.split("/")[-1]

            row[f"{short_name}_records"] = len(records)
            row[f"{short_name}_valid"] = len(valid_data)

            row[f"{short_name}_valid_pct"] = round(
                100 * len(valid_data) / len(records),
                2
            ) if len(records) else 0

            valid_times = valid_data[
                "timestamp"
            ].to_numpy()

            intervals = np.diff(valid_times)

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

        # ---------------------------------------------
        # Original recording duration
        # ---------------------------------------------

        if all_original_timestamps:

            original_start = min(all_original_timestamps)
            original_end = max(all_original_timestamps)

            row["original_recording_duration_hours"] = round(
                (original_end - original_start) / 3600,
                3
            )

        else:
            raise ValueError("No original timestamps found.")

        # ---------------------------------------------
        # MAP reference timeline
        # ---------------------------------------------

        map_name = "Solar8000/ART_MBP"

        reference = track_data[map_name]

        map_times = reference["timestamp"].to_numpy()

        row["valid_MAP_observations"] = len(map_times)

        if len(map_times):
            row["valid_MAP_span_hours"] = round(
                (map_times[-1] - map_times[0]) / 3600,
                3
            )
        else:
            raise ValueError("No valid MAP observations.")

        complete_mask = np.ones(
            len(map_times), dtype=bool
        )

        total_repeated_matches = 0

        # ---------------------------------------------
        # One-to-one alignment
        # ---------------------------------------------

        for track_name in TRACKS:

            if track_name == map_name:
                continue

            short_name = track_name.split("/")[-1]

            source_times = track_data[
                track_name
            ]["timestamp"].to_numpy()

            matched_indices, differences = match_one_to_one(
                map_times,
                source_times,
                TOLERANCE_SECONDS
            )

            matched_mask = matched_indices >= 0

            complete_mask &= matched_mask

            # A source index can only be assigned once.
            repeated_matches = int(
                len(matched_indices[matched_mask])
                - len(np.unique(matched_indices[matched_mask]))
            )

            total_repeated_matches += repeated_matches

            row[f"{short_name}_match_pct"] = round(
                100 * matched_mask.sum() / len(map_times),
                2
            )

            row[f"{short_name}_repeated_matches"] = (
                repeated_matches
            )

            row[f"{short_name}_median_time_difference"] = (
                round(float(np.nanmedian(differences)), 4)
                if matched_mask.any() else np.nan
            )

        # ---------------------------------------------
        # Complete five-signal coverage
        # ---------------------------------------------

        complete_times = map_times[complete_mask]

        row["complete_five_signal_observations"] = len(
            complete_times
        )

        row["complete_five_signal_pct"] = round(
            100 * len(complete_times) / len(map_times),
            2
        )

        row["total_repeated_matches"] = (
            total_repeated_matches
        )

        # ---------------------------------------------
        # Continuous complete-data segments
        # ---------------------------------------------

        segment_count, longest_segment = (
            get_continuous_segment_stats(
                complete_times,
                CONTINUITY_GAP_SECONDS
            )
        )

        row["complete_data_segment_count"] = segment_count

        row["longest_complete_segment_minutes"] = round(
            longest_segment / 60,
            2
        )

        row["status"] = "SUCCESS"

        print(
            "Original duration:",
            row["original_recording_duration_hours"],
            "hours"
        )

        print(
            "Valid MAP span:",
            row["valid_MAP_span_hours"],
            "hours"
        )

        print(
            "Complete five-signal coverage:",
            row["complete_five_signal_pct"],
            "%"
        )

        print(
            "Longest complete segment:",
            row["longest_complete_segment_minutes"],
            "minutes"
        )

        print(
            "Repeated matches:",
            row["total_repeated_matches"]
        )

    except Exception as error:

        row["error"] = str(error)

        print("ERROR:", error)

    results.append(row)


# =====================================================
# STEP 3: SAVE REPORT
# =====================================================

report = pd.DataFrame(results)

Path(OUTPUT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True
)

report.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== VALIDATION SUMMARY ==========")

summary_columns = [
    "caseid",
    "status",
    "original_recording_duration_hours",
    "valid_MAP_span_hours",
    "complete_five_signal_pct",
    "longest_complete_segment_minutes",
    "total_repeated_matches"
]

print(
    report[
        [
            col for col in summary_columns
            if col in report.columns
        ]
    ].to_string(index=False)
)

print(
    "\nSuccessful cases:",
    (report["status"] == "SUCCESS").sum()
)

print(
    "Failed cases:",
    (report["status"] != "SUCCESS").sum()
)

print("\nReport saved:", OUTPUT_FILE)
print("\nART cohort validation V2 completed.")
