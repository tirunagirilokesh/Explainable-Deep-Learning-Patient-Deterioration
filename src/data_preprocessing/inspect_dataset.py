
import pandas as pd

# Load datasets
cases = pd.read_csv("data/raw/cases.csv")
tracks = pd.read_csv("data/raw/tracks.csv")
labs = pd.read_csv("data/raw/labs.csv")

print("\n========== CASES DATASET ==========")
print("Shape:", cases.shape)
print("\nPatient information:")
print(cases[["age", "sex", "height", "weight", "bmi", "asa"]].head())

print("\n========== PHYSIOLOGICAL TRACKS ==========")
print("Total track records:", len(tracks))

# Identify important physiological signals
keywords = [
    "ART", "ABP", "NIBP", "MAP", "HR",
    "SPO2", "PLETH", "ECG", "RESP", "RR"
]

for keyword in keywords:
    matched = tracks[
        tracks["tname"].str.contains(
            keyword, case=False, na=False
        )
    ]["tname"].drop_duplicates().tolist()

    print(f"\n{keyword}:")
    print(matched[:30])

print("\n========== LABORATORY DATA ==========")
print("Total laboratory records:", len(labs))

print("\nMost common laboratory tests:")
print(labs["name"].value_counts().head(30))

print("\n========== MISSING VALUES ==========")
print("\nCases:")
print(cases.isnull().mean().sort_values(ascending=False).head(15))

print("\nLabs:")
print(labs.isnull().mean())
