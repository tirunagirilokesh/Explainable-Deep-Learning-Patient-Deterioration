
import os
import glob
import pandas as pd
import numpy as np

INPUT_DIR = "data/processed/extracted_cases"

MAP_COL = "Solar8000/ART_MBP"

INTERVAL = 2
MIN_EVENT_SAMPLES = 30  # 60 seconds

files = glob.glob(
    os.path.join(INPUT_DIR, "case_*.csv")
)

results = []

for i, file in enumerate(files, start=1):

    try:
        df = pd.read_csv(file)

        map_values = pd.to_numeric(
            df[MAP_COL],
            errors="coerce"
        )

        # Preliminary valid MAP range
        valid_map = map_values.between(20, 200)

        # Genuine candidate hypotension readings
        low_map = (
            valid_map &
            (map_values < 65)
        )

        # Detect consecutive low MAP samples
        groups = (
            low_map.ne(low_map.shift())
        ).cumsum()

        run_lengths = (
            low_map[low_map]
            .groupby(groups[low_map])
            .size()
        )

        sustained_events = int(
            (run_lengths >= MIN_EVENT_SAMPLES).sum()
        )

        results.append({
            "caseid": int(
                os.path.basename(file)
                .replace("case_", "")
                .replace(".csv", "")
            ),
            "valid_MAP_samples": int(valid_map.sum()),
            "low_MAP_samples": int(low_map.sum()),
            "sustained_event_runs": sustained_events,
            "has_sustained_event": sustained_events > 0
        })

    except Exception as e:
        print("Error:", file, e)

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)}")

report = pd.DataFrame(results)

output_path = (
    "data/processed/hypotension_event_report.csv"
)

report.to_csv(output_path, index=False)

print("\nHYPOTENSION CHECK COMPLETED")
print("Patients checked:", len(report))
print(
    "Patients with candidate sustained events:",
    report["has_sustained_event"].sum()
)
print(
    "Total sustained event runs:",
    report["sustained_event_runs"].sum()
)

print("\nSaved:", output_path)
