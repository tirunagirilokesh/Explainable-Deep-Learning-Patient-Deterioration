
import vitaldb
import numpy as np
import pandas as pd

FILE = "data/vital_files/case_1.vital"

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

REFERENCE_TRACK = "Solar8000/ART_MBP"
TOLERANCE_SECONDS = 1.0

vf = vitaldb.read_vital(FILE)

track_frames = {}

for name in TRACKS:
    track = vf.find_track(name)

    if track is None:
        print(f"Track not found: {name}")
        continue

    records = track.recs

    df = pd.DataFrame({
        "timestamp": [r["dt"] for r in records],
        "value": [r["val"] for r in records]
    })

    df = df[
        np.isfinite(df["value"]) &
        (df["value"] > 0)
    ].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    track_frames[name] = df

reference = track_frames[REFERENCE_TRACK].copy()

reference = reference.rename(
    columns={"value": "MAP"}
)

print("\n========== TIMESTAMP ALIGNMENT ANALYSIS ==========")
print("Reference observations (valid MAP):", len(reference))
print("Matching tolerance:", TOLERANCE_SECONDS, "seconds")

alignment = reference[["timestamp", "MAP"]].copy()

for name in TRACKS:
    if name == REFERENCE_TRACK:
        continue

    if name not in track_frames:
        continue

    short_name = name.split("/")[-1]

    source = track_frames[name].rename(
        columns={
            "timestamp": f"{short_name}_timestamp",
            "value": short_name
        }
    )
    matched = pd.merge_asof(
        alignment[["timestamp"]].sort_values("timestamp"),
        source.sort_values(f"{short_name}_timestamp"),
        left_on="timestamp",
        right_on=f"{short_name}_timestamp",
        direction="nearest",
        tolerance=TOLERANCE_SECONDS
    )

    alignment[short_name] = matched[short_name].values

    alignment[f"{short_name}_matched_timestamp"] = (
        matched[f"{short_name}_timestamp"].values
    )

    alignment[f"{short_name}_time_difference"] = (
        matched[f"{short_name}_timestamp"] -
        matched["timestamp"]
    ).abs().values

print("\n========== MATCHING RESULTS ==========")

for name in TRACKS:
    short_name = name.split("/")[-1]

    if short_name == "ART_MBP":
        continue

    if short_name not in alignment.columns:
        continue

    matched_count = alignment[short_name].notna().sum()
    unmatched_count = len(alignment) - matched_count

    print(f"\n{short_name}")
    print("Matched observations:", matched_count)
    print("Unmatched observations:", unmatched_count)
    print(
        "Match percentage:",
        round(100 * matched_count / len(alignment), 2)
    )

    differences = alignment[
        f"{short_name}_time_difference"
    ].dropna()

    if len(differences):
        print(
            "Median time difference:",
            round(differences.median(), 4)
        )
        print(
            "95th percentile time difference:",
            round(differences.quantile(0.95), 4)
        )

        matched_timestamps = alignment.loc[
        alignment[short_name].notna(),
        f"{short_name}_matched_timestamp"
    ]

    print(
        "Repeated source timestamps:",
        int(matched_timestamps.duplicated().sum())
    )

signal_columns = [
    name.split("/")[-1]
    for name in TRACKS
    if name != REFERENCE_TRACK
    and name.split("/")[-1] in alignment.columns
]

alignment["all_signals_matched"] = (
    alignment[signal_columns].notna().all(axis=1)
)

print("\n========== COMPLETE FIVE-SIGNAL COVERAGE ==========")
print(
    "All five signals matched:",
    int(alignment["all_signals_matched"].sum())
)
print(
    "Complete coverage percentage:",
    round(
        100 * alignment["all_signals_matched"].mean(),
        2
    )
)

OUTPUT = (
    "data/processed/"
    "case_1_timestamp_alignment_diagnostics.csv"
)

alignment.to_csv(OUTPUT, index=False)

print("\nDiagnostic file saved:", OUTPUT)
print("\nTimestamp alignment analysis completed.")
