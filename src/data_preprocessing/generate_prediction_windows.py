
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "processed" / "extracted_cases"

PROCESSED_DIR = ROOT / "data" / "processed"

QUALITY_FILE = PROCESSED_DIR / "continuous_signal_validation.csv"

EVENT_FILE = PROCESSED_DIR / "validated_hypotension_events.csv"

OUTPUT_DIR = PROCESSED_DIR / "windowed_dataset"

SIGNALS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

SIGNAL_LIMITS = {
    "Solar8000/ART_MBP": (20, 200),
    "Solar8000/ART_SBP": (40, 300),
    "Solar8000/ART_DBP": (10, 200),
    "Solar8000/HR": (20, 250),
    "Solar8000/PLETH_SPO2": (50, 100),
}

MAP_COL = "Solar8000/ART_MBP"

INTERVAL = 2

INPUT_SAMPLES = 900

# One new window every 60 seconds.
STRIDE_SAMPLES = 30

PRIMARY_HORIZON = 1800

EXTENDED_HORIZON = 3600

MIN_SIGNAL_VALIDITY = 0.90

MAX_INTERPOLATION_GAP = 5

RANDOM_STATE = 42

# =====================================================
# PATIENT SELECTION
# =====================================================

def select_patients():

    quality = pd.read_csv(QUALITY_FILE)

    events = pd.read_csv(EVENT_FILE)

    quality["caseid"] = pd.to_numeric(
        quality["caseid"],
        errors="coerce"
    )

    events["caseid"] = pd.to_numeric(
        events["caseid"],
        errors="coerce"
    )

    validity_columns = [
        f"{signal}_valid_pct"
        for signal in SIGNALS
    ]

    # Every signal must have at least 90% validity.
    valid_patients = quality[
        quality[validity_columns].min(axis=1)
        >= MIN_SIGNAL_VALIDITY * 100
    ].copy()

    # Require at least 60 minutes of recording.
    valid_patients = valid_patients[
        valid_patients["recording_duration_minutes"] >= 60
    ].copy()

    valid_patients["caseid"] = (
        valid_patients["caseid"].astype(int)
    )

    event_patient_ids = set(
        events["caseid"].dropna().astype(int)
    )

    valid_patients["has_event"] = (
        valid_patients["caseid"].isin(event_patient_ids)
    ).astype(int)

    print("\nPATIENT SELECTION")
    print("-----------------")
    print("Eligible patients:", len(valid_patients))
    print(
        "Patients with events:",
        int(valid_patients["has_event"].sum())
    )
    print(
        "Patients without events:",
        int((valid_patients["has_event"] == 0).sum())
    )

    if len(valid_patients) < 10:
        raise ValueError("Insufficient eligible patients.")

    return valid_patients, events


# =====================================================
# PATIENT-LEVEL SPLITTING
# =====================================================

def create_patient_splits(patients):

    patient_ids = patients["caseid"].to_numpy()

    stratify_labels = patients["has_event"].to_numpy()

    train_ids, remaining_ids = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    remaining_labels = (
        patients.set_index("caseid")
        .loc[remaining_ids, "has_event"]
        .to_numpy()
    )

    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remaining_labels
    )

    splits = {
        "train": set(train_ids),
        "validation": set(val_ids),
        "test": set(test_ids)
    }

    split_rows = []

    for split_name, ids in splits.items():

        for case_id in sorted(ids):

            patient_row = patients[
                patients["caseid"] == case_id
            ].iloc[0]

            split_rows.append({
                "caseid": case_id,
                "split": split_name,
                "has_event": int(
                    patient_row["has_event"]
                )
            })

    split_df = pd.DataFrame(split_rows)

    split_df.to_csv(
        OUTPUT_DIR / "patient_split.csv",
        index=False
    )

    print("\nPATIENT SPLIT")
    print("-------------")

    for name, ids in splits.items():
        print(f"{name}: {len(ids)} patients")

    return splits


# =====================================================
# PREPARE SIGNAL WINDOW
# =====================================================

def prepare_window(window):

    data = window[SIGNALS].copy()

    for signal, (lower, upper) in SIGNAL_LIMITS.items():

        data[signal] = pd.to_numeric(
            data[signal],
            errors="coerce"
        )

        valid = (
            np.isfinite(data[signal])
            & data[signal].between(lower, upper)
        )

        data.loc[~valid, signal] = np.nan

    # Require >=90% valid observations in each signal.
    validity = data.notna().mean()

    if (validity < MIN_SIGNAL_VALIDITY).any():
        return None

    # Interpolate only short internal gaps.
    data = data.interpolate(
        method="linear",
        limit=MAX_INTERPOLATION_GAP,
        limit_area="inside"
    )

    # Reject windows with unresolved missing values.
    if data.isna().any().any():
        return None

    return data.to_numpy(dtype=np.float32)


# =====================================================
# GENERATE WINDOWS FOR ONE PATIENT
# =====================================================

def generate_patient_windows(file_path, case_events):

    df = pd.read_csv(file_path)

    case_id = int(
        file_path.stem.replace("case_", "")
    )

    required = ["time_seconds"] + SIGNALS

    if not all(col in df.columns for col in required):
        raise ValueError("Required signal columns missing")

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=["time_seconds"])

    df = df.sort_values("time_seconds")

    df = df.drop_duplicates(
        subset="time_seconds",
        keep="first"
    ).reset_index(drop=True)

    times = df["time_seconds"].to_numpy(dtype=float)

    if len(df) < INPUT_SAMPLES:
        return None

    # Ensure the exported 2-second grid is continuous.
    if not np.all(
        np.isclose(np.diff(times), INTERVAL)
    ):
        raise ValueError(
            "Non-continuous exported time grid"
        )

    map_values = df[MAP_COL].to_numpy(dtype=float)

    # Original validated event intervals.
    if case_events.empty:

        event_starts = np.array([], dtype=float)
        event_ends = np.array([], dtype=float)

    else:

        event_starts = case_events[
            "start_time_seconds"
        ].to_numpy(dtype=float)

        event_ends = case_events[
            "end_time_seconds"
        ].to_numpy(dtype=float)

    X = []

    y30 = []
    y60 = []

    anchors = []
    future_event_times = []

    # End index of the historical window.
    for end_idx in range(
        INPUT_SAMPLES - 1,
        len(df),
        STRIDE_SAMPLES
    ):

        anchor = times[end_idx]

        horizon30_end = anchor + PRIMARY_HORIZON

        horizon60_end = anchor + EXTENDED_HORIZON

        # A complete 30-minute follow-up is mandatory.
        if times[-1] < horizon30_end:
            continue

        # Current MAP must be valid and >=65.
        current_map = map_values[end_idx]

        if (
            not np.isfinite(current_map)
            or current_map < 65
            or current_map > 200
        ):
            continue

        # Exclude anchors during existing hypotension.
        active_event = np.any(
            (event_starts <= anchor)
            & (event_ends >= anchor)
        )

        if active_event:
            continue

        # Identify new event onsets.
        future_events30 = event_starts[
            (event_starts > anchor)
            & (event_starts <= horizon30_end)
        ]

        future_events60 = event_starts[
            (event_starts > anchor)
            & (event_starts <= horizon60_end)
        ]

        # Historical input: 900 observations.
        start_idx = end_idx - INPUT_SAMPLES + 1

        window = df.iloc[
            start_idx:end_idx + 1
        ]

        prepared = prepare_window(window)

        if prepared is None:
            continue

        X.append(prepared)

        y30.append(
            int(len(future_events30) > 0)
        )

        # -1 indicates unavailable extended follow-up.
        if times[-1] >= horizon60_end:

            y60.append(
                int(len(future_events60) > 0)
            )

        else:

            y60.append(-1)

        anchors.append(anchor)

        if len(future_events30) > 0:

            future_event_times.append(
                float(np.min(future_events30))
            )

        else:

            future_event_times.append(np.nan)

    if not X:
        return None

    return {
        "caseid": case_id,
        "X": np.stack(X).astype(np.float32),
        "y30": np.asarray(y30, dtype=np.int8),
        "y60": np.asarray(y60, dtype=np.int8),
        "anchors": np.asarray(anchors),
        "event_times": np.asarray(
            future_event_times,
            dtype=np.float64
        )
    }


# =====================================================
# MAIN
# =====================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    patients, events = select_patients()

    splits = create_patient_splits(patients)

    metadata = []

    failures = []

    total_windows = 0

    print("\nGENERATING WINDOWS")
    print("------------------")

    for split_name, patient_ids in splits.items():

        split_dir = OUTPUT_DIR / split_name

        split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        split_events = []

        split_patients = patients[
            patients["caseid"].isin(patient_ids)
        ]

        for index, row in enumerate(
            split_patients.itertuples(index=False),
            start=1
        ):

            case_id = int(row.caseid)

            file_path = (
                INPUT_DIR / f"case_{case_id}.csv"
            )

            case_events = events[
                events["caseid"] == case_id
            ].copy()

            try:

                result = generate_patient_windows(
                    file_path,
                    case_events
                )

                if result is None:
                    continue

                patient_X = result["X"]

                patient_y30 = result["y30"]

                patient_y60 = result["y60"]

                patient_anchors = result["anchors"]

                patient_event_times = result["event_times"]

                output_file = (
                    split_dir / f"case_{case_id}.npz"
                )

                np.savez_compressed(
                    output_file,
                    X=patient_X,
                    y30=patient_y30,
                    y60=patient_y60,
                    anchors=patient_anchors,
                    event_times=patient_event_times
                )

                total_windows += len(patient_y30)

                for j in range(len(patient_y30)):

                    split_events.append({
                        "caseid": case_id,
                        "anchor_time_seconds": (
                            patient_anchors[j]
                        ),
                        "label_30min": int(
                            patient_y30[j]
                        ),
                        "label_60min": int(
                            patient_y60[j]
                        ),
                        "future_event_start_seconds": (
                            patient_event_times[j]
                        )
                    })

            except Exception as error:

                failures.append({
                    "caseid": case_id,
                    "split": split_name,
                    "error": str(error)
                })

            if index % 25 == 0 or index == len(split_patients):
                print(
                    f"{split_name}: "
                    f"{index}/{len(split_patients)} patients"
                )

        split_metadata = pd.DataFrame(split_events)

        split_metadata.to_csv(
            OUTPUT_DIR / f"{split_name}_metadata.csv",
            index=False
        )

        if not split_metadata.empty:

            print(f"\n{split_name.upper()} SUMMARY")

            print(
                "Windows:",
                len(split_metadata)
            )

            print(
                "30-min positive:",
                int(
                    (split_metadata["label_30min"] == 1).sum()
                )
            )

            print(
                "30-min negative:",
                int(
                    (split_metadata["label_30min"] == 0).sum()
                )
            )

            extended = split_metadata[
                split_metadata["label_60min"] != -1
            ]

            print(
                "60-min eligible windows:",
                len(extended)
            )

            print(
                "60-min positive:",
                int(
                    (extended["label_60min"] == 1).sum()
                )
            )

    # Save failures, if any.
    if failures:

        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "window_generation_failures.csv",
            index=False
        )

    print("\n========== FINAL SUMMARY ==========")

    print("Total windows generated:", total_windows)

    print("Patient failures:", len(failures))

    print("\nOutput directory:", OUTPUT_DIR)

    print("\nDataset generation completed.")


if __name__ == "__main__":
    main()
