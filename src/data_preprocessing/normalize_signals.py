
from pathlib import Path
import numpy as np
import pandas as pd
import shutil

# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "data" / "processed" / "windowed_dataset"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "normalized_windowed_dataset"
)

PARTITIONS = ["train", "validation", "test"]

SIGNALS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

EXPECTED_SHAPE = (900, 5)


# =====================================================
# STEP 1: CALCULATE TRAINING STATISTICS
# Using batch-combined Welford statistics
# =====================================================

def calculate_training_statistics():

    count = 0
    mean = np.zeros(5, dtype=np.float64)
    M2 = np.zeros(5, dtype=np.float64)

    train_dir = INPUT_DIR / "train"

    files = sorted(train_dir.glob("case_*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No training NPZ files found: {train_dir}"
        )

    print("\nCALCULATING TRAINING STATISTICS")
    print("--------------------------------")

    for index, file_path in enumerate(files, start=1):

        with np.load(file_path, allow_pickle=False) as data:

            X = data["X"]

        if X.ndim != 3 or X.shape[1:] != EXPECTED_SHAPE:
            raise ValueError(
                f"Invalid input shape in {file_path.name}: {X.shape}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite values in {file_path.name}"
            )

        batch = X.reshape(-1, 5).astype(np.float64)

        batch_count = len(batch)
        batch_mean = batch.mean(axis=0)

        batch_M2 = (
            (batch - batch_mean) ** 2
        ).sum(axis=0)

        if count == 0:

            mean = batch_mean
            M2 = batch_M2
            count = batch_count

        else:

            combined_count = count + batch_count

            delta = batch_mean - mean

            mean += delta * batch_count / combined_count

            M2 += (
                batch_M2
                + delta ** 2
                * count
                * batch_count
                / combined_count
            )

            count = combined_count

        if index % 25 == 0 or index == len(files):

            print(
                f"Processed training patients: "
                f"{index}/{len(files)}"
            )

    if count == 0:
        raise ValueError("No training observations found.")

    variance = M2 / count
    std = np.sqrt(variance)

    if not np.isfinite(mean).all():
        raise ValueError("Invalid training means.")

    if not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError("Invalid or near-zero standard deviation.")

    print("\nTRAINING STATISTICS")
    print("-------------------")

    for i, signal in enumerate(SIGNALS):

        print(f"\n{signal}")
        print(f"Mean: {mean[i]:.6f}")
        print(f"Std:  {std[i]:.6f}")

    print("\nTotal training observations:", count)

    return mean, std, count


# =====================================================
# STEP 2: NORMALIZE EACH PARTITION
# =====================================================

def normalize_partition(partition, mean, std):

    input_folder = INPUT_DIR / partition
    output_folder = OUTPUT_DIR / partition

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(input_folder.glob("case_*.npz"))

    if not files:
        raise FileNotFoundError(input_folder)

    print(f"\nNORMALIZING {partition.upper()}")
    print("------------------------------")

    total_windows = 0

    for index, file_path in enumerate(files, start=1):

        with np.load(file_path, allow_pickle=False) as data:

            arrays = {
                key: data[key]
                for key in data.files
            }

        X = arrays["X"]

        if X.ndim != 3 or X.shape[1:] != EXPECTED_SHAPE:
            raise ValueError(
                f"Incorrect shape in {file_path.name}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Invalid values in {file_path.name}"
            )

        # Apply training-derived normalization.
        X_normalized = (
            X.astype(np.float32)
            - mean.astype(np.float32).reshape(1, 1, 5)
        ) / std.astype(np.float32).reshape(1, 1, 5)

        if not np.isfinite(X_normalized).all():
            raise ValueError(
                f"Normalization produced invalid values: "
                f"{file_path.name}"
            )

        arrays["X"] = X_normalized.astype(np.float32)

        output_file = output_folder / file_path.name

        np.savez_compressed(
            output_file,
            **arrays
        )

        total_windows += len(X)

        if index % 25 == 0 or index == len(files):

            print(
                f"Processed {index}/{len(files)} patients"
            )

    print(f"{partition} windows: {total_windows}")

    return total_windows


# =====================================================
# STEP 3: SAVE NORMALIZATION INFORMATION
# =====================================================

def save_statistics(mean, std, count):

    np.savez(
        OUTPUT_DIR / "normalization_stats.npz",
        mean=mean,
        std=std,
        signals=np.asarray(SIGNALS),
        training_observations=count
    )

    report = pd.DataFrame({
        "signal": SIGNALS,
        "training_mean": mean,
        "training_std": std
    })

    report.to_csv(
        OUTPUT_DIR / "normalization_report.csv",
        index=False
    )

    # Preserve split and metadata files for traceability.
    files_to_copy = [
        "patient_split.csv",
        "train_metadata.csv",
        "validation_metadata.csv",
        "test_metadata.csv"
    ]

    for filename in files_to_copy:

        source = INPUT_DIR / filename

        if source.exists():

            shutil.copy2(
                source,
                OUTPUT_DIR / filename
            )


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 65)
    print("SIGNAL NORMALIZATION PIPELINE")
    print("=" * 65)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Do not overwrite existing normalized data. "
            "Inspect it or choose a new output directory."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Calculate statistics using training data only.
    mean, std, count = calculate_training_statistics()

    # Save reusable normalization parameters.
    save_statistics(mean, std, count)

    # Apply identical parameters to all partitions.
    results = {}

    for partition in PARTITIONS:

        results[partition] = normalize_partition(
            partition,
            mean,
            std
        )

    # Final summary.
    print("\n" + "=" * 65)
    print("NORMALIZATION SUMMARY")
    print("=" * 65)

    for partition, windows in results.items():

        print(f"{partition}: {windows} windows")

    print(
        "\nTotal normalized windows:",
        sum(results.values())
    )

    print(
        "\nNormalization parameters saved:",
        OUTPUT_DIR / "normalization_stats.npz"
    )

    print("\nSIGNAL NORMALIZATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
