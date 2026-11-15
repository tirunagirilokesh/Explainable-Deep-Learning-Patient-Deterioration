
import os
import numpy as np
import pandas as pd
import vitaldb


# ============================================================
# CONFIGURATION
# ============================================================

CASE_ID = 1

VITAL_FILE = f"data/vital_files/case_{CASE_ID}.vital"

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

MATCH_TOLERANCE_SECONDS = 1.0

GAP_THRESHOLDS = [3, 5, 10, 30, 60]

OUTPUT_DIR = "data/processed"


# ============================================================
# EXTRACT TRACK DATA
# ============================================================

def extract_track_data(vital_file, track_name):

    track = vital_file.find_track(track_name)

    if track is None:
        print(f"WARNING: Track not found: {track_name}")
        return None

    records = track.recs

    if records is None or len(records) == 0:
        print(f"WARNING: No records found: {track_name}")
        return None

    # VitalDB records are dictionaries containing dt and val.
    try:
        timestamps = np.asarray(
            [record["dt"] for record in records],
            dtype=float
        )

        values = np.asarray(
            [record["val"] for record in records],
            dtype=float
        )

    except (KeyError, TypeError, ValueError) as error:
        print(f"ERROR extracting {track_name}: {error}")
        print("First record:", repr(records[0]))
        return None

    original_count = len(records)

    # Remove non-finite timestamps and values.
    valid_mask = (
        np.isfinite(timestamps)
        & np.isfinite(values)
    )

    timestamps = timestamps[valid_mask]
    values = values[valid_mask]

    finite_count = len(timestamps)

    # Sort chronologically.
    sort_indices = np.argsort(timestamps)

    timestamps = timestamps[sort_indices]
    values = values[sort_indices]

    # Count duplicate timestamps.
    duplicate_count = int(
        pd.Series(timestamps).duplicated().sum()
    )

    # Remove duplicate timestamps, retaining the first value.
    if len(timestamps) > 0:

        unique_mask = np.concatenate(
            ([True], np.diff(timestamps) > 0)
        )

        timestamps = timestamps[unique_mask]
        values = values[unique_mask]

    return {
        "track_name": track_name,
        "original_records": original_count,
        "finite_records": finite_count,
        "duplicate_count": duplicate_count,
        "timestamps": timestamps,
        "values": values,
        "sample_rate": getattr(track, "srate", None),
        "unit": getattr(track, "unit", ""),
    }


# ============================================================
# SAMPLING INTERVAL STATISTICS
# ============================================================

def calculate_sampling_statistics(timestamps):

    if len(timestamps) < 2:
        return None

    intervals = np.diff(timestamps)

    positive_intervals = intervals[intervals > 0]

    if len(positive_intervals) == 0:
        return None

    stats = {
        "unique_records": len(timestamps),

        "time_span_seconds": (
            timestamps[-1] - timestamps[0]
        ),

        "minimum_interval": np.min(positive_intervals),

        "median_interval": np.median(positive_intervals),

        "mean_interval": np.mean(positive_intervals),

        "p95_interval": np.percentile(
            positive_intervals, 95
        ),

        "maximum_interval": np.max(positive_intervals),

        "estimated_median_frequency_hz": (
            1.0 / np.median(positive_intervals)
        ),
    }

    for threshold in GAP_THRESHOLDS:

        stats[f"gaps_over_{threshold}s"] = int(
            np.sum(positive_intervals > threshold)
        )

    return stats


# ============================================================
# TIMESTAMP MATCHING
# ============================================================

def match_timestamps(
    reference_timestamps,
    target_timestamps,
    tolerance
):

    if (
        len(reference_timestamps) == 0
        or len(target_timestamps) == 0
    ):

        return (
            np.zeros(len(reference_timestamps), dtype=bool),
            np.full(len(reference_timestamps), np.nan)
        )

    insertion_indices = np.searchsorted(
        target_timestamps,
        reference_timestamps
    )

    left_indices = np.clip(
        insertion_indices - 1,
        0,
        len(target_timestamps) - 1
    )

    right_indices = np.clip(
        insertion_indices,
        0,
        len(target_timestamps) - 1
    )

    left_diff = np.abs(
        reference_timestamps
        - target_timestamps[left_indices]
    )

    right_diff = np.abs(
        reference_timestamps
        - target_timestamps[right_indices]
    )

    nearest_diff = np.minimum(left_diff, right_diff)

    matched_mask = nearest_diff <= tolerance

    return matched_mask, nearest_diff


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("VITALDB ORIGINAL SAMPLING ANALYSIS")
    print("=" * 75)

    if not os.path.exists(VITAL_FILE):
        raise FileNotFoundError(
            f"VitalDB file not found: {VITAL_FILE}"
        )

    print("\nCase ID:", CASE_ID)
    print("VitalDB file:", VITAL_FILE)

    print(
        "Matching tolerance:",
        MATCH_TOLERANCE_SECONDS,
        "seconds"
    )

    # --------------------------------------------------------
    # LOAD VITALDB FILE
    # --------------------------------------------------------

    print("\nLoading VitalDB case...")

    vital_file = vitaldb.VitalFile(VITAL_FILE)

    print("VitalDB case loaded successfully.")

    available_tracks = vital_file.get_track_names()

    # --------------------------------------------------------
    # TRACK EXTRACTION
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TRACK EXTRACTION")
    print("=" * 75)

    track_data = {}

    for track_name in TRACKS:

        print(f"\nProcessing: {track_name}")

        if track_name not in available_tracks:
            print("Track not available.")
            continue

        result = extract_track_data(
            vital_file,
            track_name
        )

        if result is None:
            continue

        track_data[track_name] = result

        print("Original records:", result["original_records"])
        print("Finite records:", result["finite_records"])
        print("Duplicate timestamps:", result["duplicate_count"])
        print("Unique records:", len(result["timestamps"]))
        print("Sample rate metadata:", result["sample_rate"])
        print("Unit:", result["unit"])

        if len(result["timestamps"]) > 0:

            print(
                "First timestamp:",
                result["timestamps"][0]
            )

            print(
                "Last timestamp:",
                result["timestamps"][-1]
            )

            print(
                "First value:",
                result["values"][0]
            )

    if not track_data:
        raise RuntimeError("No usable tracks were extracted.")

    # --------------------------------------------------------
    # SAMPLING INTERVAL ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("ORIGINAL SAMPLING INTERVAL ANALYSIS")
    print("=" * 75)

    summary_rows = []

    for track_name, data in track_data.items():

        stats = calculate_sampling_statistics(
            data["timestamps"]
        )

        if stats is None:
            print(f"Insufficient records: {track_name}")
            continue

        print(f"\nTrack: {track_name}")
        print("-" * 60)

        for key, value in stats.items():

            if isinstance(value, (int, np.integer)):
                print(f"{key}: {value}")

            else:
                print(f"{key}: {value:.4f}")

        print("\nRecording gaps:")

        for threshold in GAP_THRESHOLDS:

            print(
                f"Gaps > {threshold} seconds:",
                stats[f"gaps_over_{threshold}s"]
            )

        summary_rows.append({
            "track": track_name,
            "original_records": data["original_records"],
            "finite_records": data["finite_records"],
            "duplicate_timestamps": data["duplicate_count"],
            "sample_rate_metadata": data["sample_rate"],
            "unit": data["unit"],
            **stats,
        })

    # --------------------------------------------------------
    # CROSS-SIGNAL TIMESTAMP MATCHING
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CROSS-SIGNAL TIMESTAMP MATCHING")
    print("=" * 75)

    reference_track = "Solar8000/ART_MBP"

    matching_rows = []

    if reference_track not in track_data:

        print("ART_MBP is unavailable as the reference.")

    else:

        reference_timestamps = track_data[
            reference_track
        ]["timestamps"]

        print("\nReference track:", reference_track)

        print(
            "Reference records:",
            len(reference_timestamps)
        )

        for track_name, data in track_data.items():

            if track_name == reference_track:
                continue

            target_timestamps = data["timestamps"]

            matched_mask, differences = match_timestamps(
                reference_timestamps,
                target_timestamps,
                MATCH_TOLERANCE_SECONDS
            )

            matched_count = int(np.sum(matched_mask))

            coverage = (
                matched_count / len(reference_timestamps) * 100
            )

            matched_differences = differences[matched_mask]

            if len(matched_differences) > 0:

                median_difference = np.median(
                    matched_differences
                )

                p95_difference = np.percentile(
                    matched_differences, 95
                )

                maximum_difference = np.max(
                    matched_differences
                )

            else:

                median_difference = np.nan
                p95_difference = np.nan
                maximum_difference = np.nan

            print(f"\nTrack: {track_name}")
            print("-" * 60)

            print("Target records:", len(target_timestamps))
            print("Matched records:", matched_count)

            print(
                "Reference coverage:",
                f"{coverage:.2f}%"
            )

            print(
                "Median timestamp difference:",
                f"{median_difference:.4f}"
                if np.isfinite(median_difference)
                else "N/A"
            )

            print(
                "95th percentile timestamp difference:",
                f"{p95_difference:.4f}"
                if np.isfinite(p95_difference)
                else "N/A"
            )

            print(
                "Maximum timestamp difference:",
                f"{maximum_difference:.4f}"
                if np.isfinite(maximum_difference)
                else "N/A"
            )

            matching_rows.append({
                "reference_track": reference_track,
                "target_track": track_name,
                "reference_records": len(reference_timestamps),
                "target_records": len(target_timestamps),
                "matched_records": matched_count,
                "coverage_percentage": coverage,
                "median_timestamp_difference": median_difference,
                "p95_timestamp_difference": p95_difference,
                "maximum_timestamp_difference": maximum_difference,
                "tolerance_seconds": MATCH_TOLERANCE_SECONDS,
            })

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"case_{CASE_ID}_sampling_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    matching_df = pd.DataFrame(matching_rows)

    matching_path = os.path.join(
        OUTPUT_DIR,
        f"case_{CASE_ID}_timestamp_matching.csv"
    )

    matching_df.to_csv(
        matching_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("REPORTS SAVED")
    print("=" * 75)

    print("\nSampling summary:")
    print(summary_path)

    print("\nTimestamp matching:")
    print(matching_path)

    print("\nAnalysis completed successfully.")

    print(
        "\nNOTE: This script evaluates recording cadence "
        "and timestamp alignment, not physiological validity."
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
