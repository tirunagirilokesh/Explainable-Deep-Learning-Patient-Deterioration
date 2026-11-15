
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DIR = ROOT / "data/processed/windowed_dataset"

NORMALIZED_DIR = (
    ROOT / "data/processed/normalized_windowed_dataset"
)

PARTITIONS = ["train", "validation", "test"]

EXPECTED_WINDOWS = {
    "train": 19441,
    "validation": 4151,
    "test": 4012
}

TOLERANCE = 0.02


def main():

    print("=" * 65)
    print("NORMALIZED DATASET VALIDATION")
    print("=" * 65)

    stats_path = NORMALIZED_DIR / "normalization_stats.npz"

    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    with np.load(stats_path, allow_pickle=False) as stats:

        mean = stats["mean"]
        std = stats["std"]
        signals = stats["signals"]

    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std > 0)

    print("\nNormalization parameters loaded.")
    print("Signals:", signals.tolist())

    total_windows = 0

    train_count = 0
    train_mean = np.zeros(5, dtype=np.float64)
    train_M2 = np.zeros(5, dtype=np.float64)

    # ------------------------------------------------
    # PARTITION VALIDATION
    # ------------------------------------------------

    for partition in PARTITIONS:

        print(f"\nVALIDATING {partition.upper()}")
        print("-" * 45)

        original_folder = ORIGINAL_DIR / partition
        normalized_folder = NORMALIZED_DIR / partition

        files = sorted(
            normalized_folder.glob("case_*.npz")
        )

        original_files = sorted(
            original_folder.glob("case_*.npz")
        )

        if not files:
            raise ValueError(f"No files in {partition}")

        if {f.name for f in files} != {
            f.name for f in original_files
        }:
            raise ValueError(
                f"Patient file mismatch in {partition}"
            )

        partition_windows = 0

        for index, file_path in enumerate(files, start=1):

            original_path = original_folder / file_path.name

            with np.load(file_path, allow_pickle=False) as norm:
                X = norm["X"]

                normalized_arrays = {
                    key: norm[key]
                    for key in norm.files
                }

            with np.load(original_path, allow_pickle=False) as orig:

                original_X = orig["X"]

                # Check unchanged labels and timestamps.
                for key in [
                    "y30",
                    "y60",
                    "anchors",
                    "event_times"
                ]:

                    if not np.array_equal(
                        orig[key],
                        normalized_arrays[key],
                        equal_nan=True
                    ):
                        raise ValueError(
                            f"{key} changed in {file_path.name}"
                        )

            # Shape validation.
            if X.ndim != 3 or X.shape[1:] != (900, 5):
                raise ValueError(
                    f"Incorrect shape: {file_path.name}"
                )

            if not np.isfinite(X).all():
                raise ValueError(
                    f"NaN or infinity in {file_path.name}"
                )

            if not np.allclose(
                X,
                (original_X - mean.reshape(1, 1, 5))
                / std.reshape(1, 1, 5),
                atol=2e-5,
                rtol=2e-5
            ):
                raise ValueError(
                    f"Incorrect normalization: {file_path.name}"
                )

            partition_windows += len(X)

            # Calculate aggregate training statistics.
            if partition == "train":

                batch = X.reshape(-1, 5).astype(np.float64)

                batch_count = len(batch)
                batch_mean = batch.mean(axis=0)

                batch_M2 = (
                    (batch - batch_mean) ** 2
                ).sum(axis=0)

                if train_count == 0:

                    train_count = batch_count
                    train_mean = batch_mean
                    train_M2 = batch_M2

                else:

                    combined = train_count + batch_count

                    delta = batch_mean - train_mean

                    train_M2 += (
                        batch_M2
                        + delta ** 2
                        * train_count
                        * batch_count
                        / combined
                    )

                    train_mean += (
                        delta * batch_count / combined
                    )

                    train_count = combined

            if index % 25 == 0 or index == len(files):
                print(
                    f"Validated {index}/{len(files)} patients"
                )

        if partition_windows != EXPECTED_WINDOWS[partition]:
            raise ValueError(
                f"Window count mismatch in {partition}"
            )

        metadata_path = (
            NORMALIZED_DIR / f"{partition}_metadata.csv"
        )

        original_metadata_path = (
            ORIGINAL_DIR / f"{partition}_metadata.csv"
        )

        normalized_metadata = pd.read_csv(metadata_path)
        original_metadata = pd.read_csv(original_metadata_path)

        if not normalized_metadata.equals(original_metadata):
            raise ValueError(
                f"Metadata changed in {partition}"
            )

        print("Windows:", partition_windows)
        print("Metadata: PASSED")
        print("Normalization: PASSED")

        total_windows += partition_windows

    # ------------------------------------------------
    # TRAINING STATISTICS
    # ------------------------------------------------

    actual_std = np.sqrt(train_M2 / train_count)

    print("\n" + "=" * 65)
    print("NORMALIZED TRAINING STATISTICS")
    print("=" * 65)

    for i, signal in enumerate(signals):

        print(f"\n{signal}")
        print(f"Mean: {train_mean[i]:.6f}")
        print(f"Std:  {actual_std[i]:.6f}")

        if abs(train_mean[i]) > TOLERANCE:
            raise ValueError("Training mean outside tolerance")

        if abs(actual_std[i] - 1.0) > TOLERANCE:
            raise ValueError("Training std outside tolerance")

    # ------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    print("Total normalized windows:", total_windows)
    print("Training observations:", train_count)

    print("\nALL NORMALIZED DATA VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
