
import pandas as pd

tracks = pd.read_csv("data/raw/tracks.csv")
cases = pd.read_csv("data/raw/cases.csv")

# Required physiological channels
required_tracks = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/RR"
]

print("\n========== TRACK AVAILABILITY ==========")

coverage = tracks.groupby("tname").agg(
    unique_cases=("caseid", "nunique"),
    track_records=("tid", "count")
).reset_index()

for track in required_tracks:
    result = coverage[coverage["tname"] == track]

    if not result.empty:
        print(f"\nTrack: {track}")
        print(result.to_string(index=False))
    else:
        print(f"\nTrack not found: {track}")

# Check patients having all required tracks
available = tracks[
    tracks["tname"].isin(required_tracks)
]

case_counts = available.groupby("caseid")["tname"].nunique()

complete_cases = case_counts[
    case_counts == len(required_tracks)
]

print("\n========== OVERALL COVERAGE ==========")
print("Total patients:", len(cases))
print("Patients with at least one required track:", len(case_counts))
print("Patients with all six tracks:", len(complete_cases))

print("\n========== SAMPLE COMPLETE CASES ==========")
print(complete_cases.head(20))

print("\n========== TRACK DISTRIBUTION ==========")
print(
    coverage[
        coverage["tname"].isin(required_tracks)
    ].sort_values("unique_cases", ascending=False)
)

# Compare invasive and non-invasive BP availability

bp_tracks = [
    "Solar8000/ART_MBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/NIBP_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

print("\n========== BP CHANNEL COMPARISON ==========")

for track in bp_tracks:
    ids = tracks.loc[
        tracks["tname"] == track, "caseid"
    ].unique()

    print(f"{track}: {len(ids)} cases")

# Calculate coverage for core channels
core_tracks = [
    "Solar8000/ART_MBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

core_cases = tracks[
    tracks["tname"].isin(core_tracks)
].groupby("caseid")["tname"].nunique()

print("\nCases with MAP + HR + SpO2:",
      (core_cases == len(core_tracks)).sum())

# Check invasive BP + HR + SpO2
# and non-invasive BP + HR + SpO2
for bp in ["Solar8000/ART_MBP", "Solar8000/NIBP_MBP"]:
    required = [bp, "Solar8000/HR", "Solar8000/PLETH_SPO2"]

    counts = tracks[
        tracks["tname"].isin(required)
    ].groupby("caseid")["tname"].nunique()

    print(f"Cases with {bp} + HR + SpO2:",
          (counts == len(required)).sum())
