
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "processed" / "extracted_cases"
OUTPUT_DIR = ROOT / "data" / "processed"

MAP_COL = "Solar8000/ART_MBP"

SIGNALS = {
    "Solar8000/ART_MBP": (20, 200),
    "Solar8000/ART_SBP": (40, 300),
    "Solar8000/ART_DBP": (10, 200),
    "Solar8000/HR": (20, 250),
    "Solar8000/PLETH_SPO2": (50, 100),
}

INTERVAL = 2.0
MAP_THRESHOLD = 65.0

# 31 observations span 60 seconds.
MIN_EVENT_SAMPLES = 31
MIN_EVENT_DURATION = 60.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DETECT VALID HYPOTENSION EVENTS
# ==========================================

def detect_events(df, case_id):

    events = []

    times = df["time_seconds"].to_numpy(dtype=float)

    map_values = df[MAP_COL].to_numpy(dtype=float)

    valid_map = (
        np.isfinite(map_values)
        & (map_values >= 20)
        & (map_values <= 200)
    )

    low_map = valid_map & (map_values < MAP_THRESHOLD)

    start = None

    def save_event(start_idx, end_idx):

        count = end_idx - start_idx + 1

        duration = times[end_idx] - times[start_idx]

        if (
            count < MIN_EVENT_SAMPLES
            or duration < MIN_EVENT_DURATION
        ):
            return

        values = map_values[start_idx:end_idx + 1]

        events.append({
            "caseid": case_id,
            "start_time_seconds": times[start_idx],
            "end_time_seconds": times[end_idx],
            "duration_seconds": duration,
            "sample_count": count,
            "mean_MAP": np.mean(values),
            "minimum_MAP": np.min(values),
            "maximum_MAP": np.max(values),
        })

    for i in range(len(df)):

        # Check whether the current observation follows
        # the previous observation on the 2-second grid.
        continuous = (
            i > 0
            and np.isclose(
                times[i] - times[i - 1],
                INTERVAL
            )
        )

        if low_map[i]:

            if start is None:
                start = i

            elif not continuous:

                save_event(start, i - 1)
                start = i

        else:

            if start is not None:
                save_event(start, i - 1)
                start = None

    # Handle event continuing until recording ends.
    if start is not None:
        save_event(start, len(df) - 1)

    return events


# ==========================================
# PATIENT SIGNAL VALIDATION
# ==========================================

def validate_patient(file_path):

    df = pd.read_csv(file_path)

    case_id = int(
        file_path.stem.replace("case_", "")
    )

    required_columns = [
        "time_seconds",
        *SIGNALS.keys()
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_columns].copy()

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=["time_seconds"])

    df = df.sort_values("time_seconds")

    duplicate_count = int(
        df["time_seconds"].duplicated().sum()
    )

    df = df.drop_duplicates(
        subset="time_seconds",
        keep="first"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid timestamps")

    times = df["time_seconds"].to_numpy(dtype=float)

    differences = np.diff(times)

    timestamp_gaps = int(
        np.sum(~np.isclose(differences, INTERVAL))
    )

    quality = {
        "caseid": case_id,
        "total_samples": len(df),
        "recording_duration_minutes": (
            times[-1] - times[0]
        ) / 60,
        "duplicate_timestamps": duplicate_count,
        "timestamp_gaps": timestamp_gaps,
    }

    for signal, (lower, upper) in SIGNALS.items():

        values = df[signal].to_numpy(dtype=float)

        valid = (
            np.isfinite(values)
            & (values >= lower)
            & (values <= upper)
        )

        quality[f"{signal}_valid_pct"] = (
            valid.mean() * 100
        )

        quality[f"{signal}_valid_count"] = int(
            valid.sum()
        )

    # Detect sustained hypotension.
    events = detect_events(df, case_id)

    quality["validated_event_count"] = len(events)

    return quality, events


# ==========================================
# MAIN
# ==========================================

def main():

    files = sorted(
        INPUT_DIR.glob("case_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No patient files found: {INPUT_DIR}"
        )

    quality_results = []
    all_events = []
    failures = []

    print("Patients found:", len(files))
    print("Starting signal validation...\n")

    for index, file_path in enumerate(files, start=1):

        try:

            quality, events = validate_patient(file_path)

            quality_results.append(quality)
            all_events.extend(events)

        except Exception as error:

            failures.append({
                "file": file_path.name,
                "error": str(error)
            })

        if index % 100 == 0 or index == len(files):
            print(f"Validated {index}/{len(files)}")

    # Save reports.
    quality_df = pd.DataFrame(quality_results)

    events_df = pd.DataFrame(
        all_events,
        columns=[
            "caseid",
            "start_time_seconds",
            "end_time_seconds",
            "duration_seconds",
            "sample_count",
            "mean_MAP",
            "minimum_MAP",
            "maximum_MAP",
        ]
    )

    quality_path = (
        OUTPUT_DIR / "continuous_signal_validation.csv"
    )

    event_path = (
        OUTPUT_DIR / "validated_hypotension_events.csv"
    )

    failure_path = (
        OUTPUT_DIR / "continuous_validation_failures.csv"
    )

    quality_df.to_csv(quality_path, index=False)
    events_df.to_csv(event_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            failure_path,
            index=False
        )

    # ======================================
    # SUMMARY
    # ======================================

    print("\n========== VALIDATION SUMMARY ==========")

    print("Patients processed:", len(quality_df))
    print("Patients failed:", len(failures))

    if not quality_df.empty:

        valid_columns = [
            f"{signal}_valid_pct"
            for signal in SIGNALS
        ]

        print(
            "\nPatients with all signals >=90% valid:",
            int(
                (
                    quality_df[valid_columns]
                    .min(axis=1) >= 90
                ).sum()
            )
        )

        print(
            "Patients with timestamp gaps:",
            int(
                (quality_df["timestamp_gaps"] > 0).sum()
            )
        )

        print(
            "Patients with duplicate timestamps:",
            int(
                (quality_df["duplicate_timestamps"] > 0).sum()
            )
        )

    print("\nValidated hypotension events:", len(events_df))

    if not events_df.empty:

        print(
            "Patients with events:",
            events_df["caseid"].nunique()
        )

        print("\nEvent duration (seconds):")
        print(events_df["duration_seconds"].describe())

        print("\nMean MAP during events:")
        print(events_df["mean_MAP"].describe())

    print("\nSaved reports:")
    print(quality_path)
    print(event_path)

    if failures:
        print(failure_path)


if __name__ == "__main__":
    main()
