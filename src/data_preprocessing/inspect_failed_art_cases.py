
import vitaldb
import pandas as pd

TRACKS_FILE = "data/raw/tracks.csv"

FAILED_CASES = [5382, 3055, 5830, 5747]

REQUIRED_TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2"
]

tracks_df = pd.read_csv(TRACKS_FILE)

print("\n========== FAILED CASE INVESTIGATION ==========")

for case_id in FAILED_CASES:

    print(f"\n{'=' * 50}")
    print(f"CASE ID: {case_id}")

    # Check metadata
    metadata_tracks = tracks_df[
        tracks_df["caseid"] == case_id
    ]["tname"].tolist()

    print("\nTracks listed in metadata:")

    for name in REQUIRED_TRACKS:
        print(
            name,
            "->",
            "AVAILABLE" if name in metadata_tracks else "MISSING"
        )

    # Retrieve actual VitalFile
    try:
        vf = vitaldb.VitalFile(case_id, REQUIRED_TRACKS)

        actual_tracks = vf.get_track_names()

        print("\nTracks retrieved from VitalFile:")

        for name in actual_tracks:
            print("-", name)

        print("\nRequired track inspection:")

        for name in REQUIRED_TRACKS:

            track = vf.find_track(name)

            if track is None:
                print(name, "-> NOT FOUND")
            else:
                print(
                    name,
                    "-> FOUND",
                    "| Records:",
                    len(track.recs)
                )

    except Exception as error:

        print("\nVitalFile loading error:")
        print(type(error).__name__, str(error))

print("\n========== INVESTIGATION COMPLETED ==========")
