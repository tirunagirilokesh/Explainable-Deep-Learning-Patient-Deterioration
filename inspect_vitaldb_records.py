
import vitaldb
import numpy as np

FILE_PATH = "data/vital_files/case_1.vital"

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

vf = vitaldb.VitalFile(FILE_PATH)

print("=" * 70)
print("VITALDB RECORD STRUCTURE INSPECTION")
print("=" * 70)

print("\nVitalDB version:")
print(getattr(vitaldb, "__version__", "Not exposed"))

for name in TRACKS:

    print("\n" + "=" * 70)
    print("TRACK:", name)

    track = vf.find_track(name)

    if track is None:
        print("Track not found")
        continue

    print("Track object type:", type(track))

    print("\nTrack attributes:")
    print(vars(track))

    recs = track.recs

    print("\nRecords type:", type(recs))
    print("Records shape:", np.asarray(recs).shape)
    print("Records dtype:", np.asarray(recs).dtype)

    print("\nFirst three records:")

    for i, rec in enumerate(recs[:3]):
        print(f"Record {i}:")
        print("Type:", type(rec))
        print("Value:", repr(rec))

    print("\nTrack sample rate:", getattr(track, "srate", "Unavailable"))
    print("Track unit:", getattr(track, "unit", "Unavailable"))
