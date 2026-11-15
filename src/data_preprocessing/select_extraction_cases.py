
import pandas as pd
from pathlib import Path

# Existing VitalDB metadata
TRACK_FILE = Path("data/raw/tracks.csv")

df = pd.read_csv(TRACK_FILE)

print("Metadata columns:", df.columns.tolist())

required_columns = {"caseid", "tname"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required_columns}, "
        f"but found {df.columns.tolist()}"
    )

# Core physiological signals
art_tracks = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

nibp_tracks = [
    "Solar8000/NIBP_MBP",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

def find_eligible_cases(track_list):
    selected = df[df["tname"].isin(track_list)]

    coverage = (
        selected.groupby("tname")["caseid"]
        .nunique()
    )

    case_sets = [
        set(selected.loc[
            selected["tname"] == track,
            "caseid"
        ])
        for track in track_list
    ]

    eligible = set.intersection(*case_sets)

    print("\nTrack coverage:")
    print(coverage)

    print("\nEligible cases:", len(eligible))

    return sorted(eligible)


print("\n========== ARTERIAL COHORT ==========")
art_cases = find_eligible_cases(art_tracks)

print("\n========== NIBP COHORT ==========")
nibp_cases = find_eligible_cases(nibp_tracks)

print("\nFirst 10 arterial case IDs:")
print(art_cases[:10])

print("\nFirst 10 NIBP case IDs:")
print(nibp_cases[:10])

print("\n========== FINAL COHORT SUMMARY ==========")

print("\nARTERIAL COHORT")
print("Total eligible cases:", len(art_cases))
print("First 10:", art_cases[:10])

print("\nNIBP COHORT")
print("Total eligible cases:", len(nibp_cases))
print("First 10:", nibp_cases[:10])

print("\n========== INDIVIDUAL TRACK COVERAGE ==========")

for track in art_tracks + nibp_tracks:
    count = df.loc[
        df["tname"] == track, "caseid"
    ].nunique()

    print(f"{track}: {count} cases")

print("\n========== COHORT VALIDATION ==========")

assert len(art_cases) <= df.loc[
    df["tname"] == "Solar8000/ART_MBP", "caseid"
].nunique()

assert len(nibp_cases) <= df.loc[
    df["tname"] == "Solar8000/NIBP_MBP", "caseid"
].nunique()

print("Cohort validation completed.")
