
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "windowed_dataset"

PARTITIONS = ["train", "validation", "test"]

EXPECTED_WINDOWS = {
    "train": 19441,
    "validation": 4151,
    "test": 4012
}

TIME_STEPS = 900
SIGNALS = 5

REQUIRED_ARRAYS = [
    "X",
    "y30",
    "y60",
    "anchors",
    "event_times"
]

PATIENT_ID_COLUMN = "caseid"


# =====================================================
# HELPER: VALIDATE NUMERICAL ARRAYS
# =====================================================

def check_numeric_quality(array, name):

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} contains non-numeric data.")

    missing = int(np.isnan(array).sum())
    infinite = int(np.isinf(array).sum())

    print(f"  {name}:")
    print(f"    Missing: {missing}")
    print(f"    Infinite: {infinite}")

    if missing > 0 or infinite > 0:
        raise ValueError(
            f"Invalid numerical values found in {name}."
        )


# =====================================================
# HELPER: VALIDATE ONE PATIENT FILE
# =====================================================

def validate_patient_file(file_path, partition):

    with np.load(file_path, allow_pickle=False) as data:

        missing_keys = [
            key for key in REQUIRED_ARRAYS
            if key not in data.files
        ]

        if missing_keys:
            raise ValueError(
                f"{file_path.name}: Missing arrays {missing_keys}"
            )

        X = data["X"]
        y30 = data["y30"]
        y60 = data["y60"]
        anchors = data["anchors"]
        event_times = data["event_times"]

    # Shape checks

    if X.ndim != 3:
        raise ValueError(
            f"{file_path.name}: X must have 3 dimensions."
        )

    if X.shape[1:] != (TIME_STEPS, SIGNALS):
        raise ValueError(
            f"{file_path.name}: Incorrect X shape {X.shape}"
        )

    n = len(X)

    for name, array in {
        "y30": y30,
        "y60": y60,
        "anchors": anchors,
        "event_times": event_times
    }.items():

        if array.ndim != 1 or len(array) != n:
            raise ValueError(
                f"{file_path.name}: {name} length mismatch."
            )

    # Numerical quality

    check_numeric_quality(X, "X")
    check_numeric_quality(anchors, "anchors")

    if not np.isfinite(event_times[~np.isnan(event_times)]).all():
        raise ValueError(
            f"{file_path.name}: Invalid event timestamps."
        )

    # Label checks

    if not set(np.unique(y30)).issubset({0, 1}):
        raise ValueError(
            f"{file_path.name}: Invalid 30-minute labels."
        )

    if not set(np.unique(y60)).issubset({-1, 0, 1}):
        raise ValueError(
            f"{file_path.name}: Invalid 60-minute labels."
        )

    # Nested horizon consistency:
    # An event within 30 minutes must also be within 60 minutes
    # whenever the 60-minute follow-up is available.

    available60 = y60 != -1

    if np.any(
        (y30 == 1)
        & available60
        & (y60 != 1)
    ):
        raise ValueError(
            f"{file_path.name}: Inconsistent 30/60-minute labels."
        )

    # Event timestamp consistency for 30-minute labels

    if np.any((y30 == 1) & np.isnan(event_times)):
        raise ValueError(
            f"{file_path.name}: Positive 30-minute label "
            "without event timestamp."
        )

    if np.any((y30 == 0) & ~np.isnan(event_times)):
        raise ValueError(
            f"{file_path.name}: Negative 30-minute label "
            "has an event timestamp."
        )

    if not np.all(np.diff(anchors) > 0):
        raise ValueError(
            f"{file_path.name}: Anchors are not strictly increasing."
        )

    # File name and patient ID

    case_id = int(file_path.stem.replace("case_", ""))

    return {
        "caseid": case_id,
        "windows": n,
        "positive30": int((y30 == 1).sum()),
        "negative30": int((y30 == 0).sum()),
        "eligible60": int(available60.sum()),
        "positive60": int((y60 == 1).sum()),
        "X": X,
        "y30": y30,
        "y60": y60,
        "anchors": anchors,
        "event_times": event_times
    }


# =====================================================
# MAIN VALIDATION
# =====================================================

def main():

    print("=" * 65)
    print("WINDOWED DATASET VALIDATION")
    print("=" * 65)

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)

    split_path = DATA_DIR / "patient_split.csv"

    if not split_path.exists():
        raise FileNotFoundError(split_path)

    patient_split = pd.read_csv(split_path)

    if PATIENT_ID_COLUMN not in patient_split.columns:
        raise ValueError(
            "caseid missing from patient_split.csv"
        )

    if "split" not in patient_split.columns:
        raise ValueError(
            "split column missing from patient_split.csv"
        )

    if patient_split["caseid"].duplicated().any():
        raise ValueError(
            "Duplicate patients in patient_split.csv"
        )

    # -------------------------------------------------
    # PATIENT SPLIT VALIDATION
    # -------------------------------------------------

    print("\n1. PATIENT SPLIT VALIDATION")

    print(
        patient_split["split"].value_counts().to_string()
    )

    if not set(patient_split["split"].unique()).issubset(
        set(PARTITIONS)
    ):
        raise ValueError("Unexpected split name detected.")

    # -------------------------------------------------
    # PARTITION VALIDATION
    # -------------------------------------------------

    all_patient_ids = {}
    total_windows = 0

    for partition in PARTITIONS:

        print("\n" + "=" * 65)
        print(f"2. VALIDATING {partition.upper()}")
        print("=" * 65)

        folder = DATA_DIR / partition

        if not folder.exists():
            raise FileNotFoundError(folder)

        files = sorted(folder.glob("case_*.npz"))

        if not files:
            raise FileNotFoundError(
                f"No patient NPZ files found in {folder}"
            )

        results = []

        for index, file_path in enumerate(files, start=1):

            result = validate_patient_file(
                file_path,
                partition
            )

            results.append(result)

            if index % 25 == 0 or index == len(files):
                print(
                    f"Validated {index}/{len(files)} patients"
                )

        patient_ids = {
            result["caseid"] for result in results
        }

        all_patient_ids[partition] = patient_ids

        partition_windows = sum(
            result["windows"] for result in results
        )

        total_windows += partition_windows

        print("\nPARTITION SUMMARY")
        print("Patients:", len(patient_ids))
        print("Windows:", partition_windows)
        print(
            "30-minute positive:",
            sum(r["positive30"] for r in results)
        )
        print(
            "30-minute negative:",
            sum(r["negative30"] for r in results)
        )
        print(
            "60-minute eligible:",
            sum(r["eligible60"] for r in results)
        )
        print(
            "60-minute positive:",
            sum(r["positive60"] for r in results)
        )

        if partition_windows != EXPECTED_WINDOWS[partition]:
            raise ValueError(
                f"{partition}: Expected "
                f"{EXPECTED_WINDOWS[partition]} windows, "
                f"found {partition_windows}."
            )

        # -------------------------------------------------
        # METADATA VALIDATION
        # -------------------------------------------------

        metadata_path = (
            DATA_DIR / f"{partition}_metadata.csv"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        metadata = pd.read_csv(metadata_path)

        required_columns = [
            "caseid",
            "anchor_time_seconds",
            "label_30min",
            "label_60min",
            "future_event_start_seconds"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing metadata columns: {missing_columns}"
            )

        if len(metadata) != partition_windows:
            raise ValueError(
                f"{partition}: Metadata row count mismatch."
            )

        metadata_ids = set(
            metadata["caseid"].dropna().astype(int).unique()
        )

        if metadata_ids != patient_ids:
            raise ValueError(
                f"{partition}: Metadata patient IDs "
                "do not match NPZ files."
            )

        # Compare metadata against the actual NPZ contents.

        for result in results:

            patient_meta = metadata[
                metadata["caseid"] == result["caseid"]
            ].reset_index(drop=True)

            if len(patient_meta) != result["windows"]:
                raise ValueError(
                    f"{partition}: Metadata window mismatch "
                    f"for patient {result['caseid']}."
                )

            comparisons = [
                (
                    patient_meta["label_30min"].to_numpy(),
                    result["y30"],
                    "label_30min"
                ),
                (
                    patient_meta["label_60min"].to_numpy(),
                    result["y60"],
                    "label_60min"
                ),
                (
                    patient_meta["anchor_time_seconds"].to_numpy(),
                    result["anchors"],
                    "anchor_time_seconds"
                ),
                (
                    patient_meta[
                        "future_event_start_seconds"
                    ].to_numpy(dtype=float),
                    result["event_times"],
                    "future_event_start_seconds"
                )
            ]

            for actual, expected, name in comparisons:

                if not np.allclose(
                    actual,
                    expected,
                    equal_nan=True
                ):
                    raise ValueError(
                        f"{partition}: Metadata mismatch "
                        f"for patient {result['caseid']}, "
                        f"column {name}."
                    )

        print("Metadata consistency: PASSED")

    # -------------------------------------------------
    # PATIENT LEAKAGE VALIDATION
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("3. PATIENT-LEVEL LEAKAGE VALIDATION")
    print("=" * 65)

    for i, first in enumerate(PARTITIONS):

        for second in PARTITIONS[i + 1:]:

            overlap = (
                all_patient_ids[first]
                & all_patient_ids[second]
            )

            print(
                f"{first} vs {second}: "
                f"{len(overlap)} overlapping patients"
            )

            if overlap:
                raise ValueError(
                    "Patient leakage detected!"
                )


    # -------------------------------------------------
    # PATIENT SPLIT CONSISTENCY
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("4. PATIENT SPLIT CONSISTENCY")
    print("=" * 65)

    for partition in PARTITIONS:

        expected_ids = set(
            patient_split.loc[
                patient_split["split"] == partition,
                "caseid"
            ].astype(int)
        )

        actual_ids = all_patient_ids[partition]

        # Every generated patient must belong to
        # the correct original partition.
        unexpected = actual_ids - expected_ids

        if unexpected:
            raise ValueError(
                f"{partition}: Patients found in incorrect "
                f"partition: {sorted(unexpected)}"
            )

        # Patients selected but without generated windows.
        no_windows = expected_ids - actual_ids

        print(f"\n{partition.upper()}")
        print("Patients assigned:", len(expected_ids))
        print("Patients with windows:", len(actual_ids))
        print("Patients without windows:", len(no_windows))

        if no_windows:
            print(
                "Patients without windows:",
                sorted(no_windows)
            )

    print("\nPatient split consistency: PASSED")


    # -------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    print("Total patients:", sum(
        len(all_patient_ids[p]) for p in PARTITIONS
    ))

    print("Total windows:", total_windows)

    for partition in PARTITIONS:
        print(
            f"{partition}: "
            f"{len(all_patient_ids[partition])} patients"
        )

    print("\nALL STRUCTURAL VALIDATION CHECKS PASSED.")

    print(
        "\nNote: Structural validation does not independently "
        "prove the clinical correctness of event labels."
    )


if __name__ == "__main__":
    main()
