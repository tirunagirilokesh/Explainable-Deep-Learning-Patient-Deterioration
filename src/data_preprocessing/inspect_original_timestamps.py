import vitaldb
import numpy as np

# ==========================================
# 1. LOAD ORIGINAL VITAL FILE
# ==========================================

FILE = "data/vital_files/case_1.vital"

TRACKS = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
]

MAP_TRACK = "Solar8000/ART_MBP"

MAP_THRESHOLD = 65.0
MIN_EVENT_DURATION = 60.0
MAX_ALLOWED_GAP = 3.0

MATCH_TOLERANCE = 1.0
CONTEXT_SECONDS = 10.0

print(f"Loading VitalDB file: {FILE}")
vf = vitaldb.read_vital(FILE)


def get_track_arrays(track):
    """Return chronologically sorted timestamp and value arrays."""
    records = getattr(track, "recs", None)

    if records is None or len(records) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if isinstance(records[0], dict):
        timestamps = np.array([record["dt"] for record in records], dtype=float)
        values = np.array([record["val"] for record in records], dtype=float)
    else:
        timestamps = np.array([record.dt for record in records], dtype=float)
        values = np.array([record.val for record in records], dtype=float)

    finite_timestamps = np.isfinite(timestamps)
    timestamps = timestamps[finite_timestamps]
    values = values[finite_timestamps]

    order = np.argsort(timestamps)
    return timestamps[order], values[order]


# Load all selected original tracks once.
signal_data = {}

for track_name in TRACKS:
    track = vf.find_track(track_name)

    if track is None:
        print(f"WARNING: Track not found: {track_name}")
        continue

    times, values = get_track_arrays(track)

    if len(times) == 0:
        print(f"WARNING: No records found: {track_name}")
        continue

    signal_data[track_name] = (times, values)


# ==========================================
# 2. INSPECT ORIGINAL TRACK RECORDS
# ==========================================

print("\n========== ORIGINAL TRACK INSPECTION ==========")

for track_name in TRACKS:
    print(f"\nTRACK: {track_name}")

    track = vf.find_track(track_name)

    if track is None:
        print("Track not found")
        continue

    records = getattr(track, "recs", None)

    if records is None:
        print("No recs attribute found")
        continue

    print("Track object type:", type(track))
    try:
        print("Available attributes:", list(vars(track).keys()))
    except TypeError:
        print("Track attributes are not available through vars().")

    print("Number of original records:", len(records))
    print("First five original records:")

    for record in records[:5]:
        print(record)

    times, values = signal_data[track_name]

    if len(times) > 1:
        intervals = np.diff(times)

        print("Timestamp intervals (seconds):")
        print("Minimum:", float(np.min(intervals)))
        print("Median:", float(np.median(intervals)))
        print("Maximum:", float(np.max(intervals)))
        print("First 10 timestamps:", times[:10])
        print("First 10 values:", values[:10])

        valid_positive = np.isfinite(values) & (values > 0)
        print("Valid positive readings:", int(np.sum(valid_positive)))
        print("Invalid or non-positive readings:", int(np.sum(~valid_positive)))


# ==========================================
# 3. MAP HYPOTENSION EVENT DETECTION
# ==========================================

print("\n========== MAP HYPOTENSION EVENT ANALYSIS ==========")

if MAP_TRACK not in signal_data:
    raise ValueError(f"Required MAP track not available: {MAP_TRACK}")

timestamps, map_values = signal_data[MAP_TRACK]

# Positive finite readings are provisionally usable.
# Very low positive readings are retained for investigation, not discarded.
valid = np.isfinite(map_values) & (map_values > 0)
below_threshold = valid & (map_values < MAP_THRESHOLD)

episodes = []
current_episode = []

for i in range(len(timestamps)):
    if not below_threshold[i]:
        if current_episode:
            episodes.append(current_episode)
            current_episode = []
        continue

    if current_episode:
        gap = timestamps[i] - timestamps[current_episode[-1]]

        if gap > MAX_ALLOWED_GAP:
            episodes.append(current_episode)
            current_episode = []

    current_episode.append(i)

if current_episode:
    episodes.append(current_episode)

event_rows = []

for episode_number, indices in enumerate(episodes, start=1):
    start_time = timestamps[indices[0]]
    end_time = timestamps[indices[-1]]
    duration = end_time - start_time
    episode_values = map_values[indices]

    event_rows.append({
        "episode": episode_number,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "number_of_readings": len(indices),
        "minimum_MAP": float(np.min(episode_values)),
        "mean_MAP": float(np.mean(episode_values)),
        "extreme_low_reading": bool(np.any(episode_values < 20)),
        "duration_qualified_candidate": duration >= MIN_EVENT_DURATION,
    })

duration_qualified = [
    event for event in event_rows
    if event["duration_qualified_candidate"]
]

shorter = [
    event for event in event_rows
    if not event["duration_qualified_candidate"]
]

print("MAP threshold:", MAP_THRESHOLD, "mmHg")
print("Minimum observed duration:", MIN_EVENT_DURATION, "seconds")
print("Maximum allowed observation gap:", MAX_ALLOWED_GAP, "seconds")
print("Total below-threshold episodes:", len(event_rows))
print("Duration-qualified candidates:", len(duration_qualified))
print("Shorter episodes:", len(shorter))
print(
    "Episodes containing MAP below 20:",
    sum(event["extreme_low_reading"] for event in event_rows),
)

if event_rows:
    print("\nAll detected episodes:")
    for event in event_rows:
        print(event)

    print("\nLongest episodes:")
    longest = sorted(
        event_rows,
        key=lambda event: event["duration_seconds"],
        reverse=True,
    )[:10]

    for event in longest:
        print(event)

print("\nMAP event detection completed.")


# ==========================================
# 4. INSPECT SELECTED MAP EPISODES
# ==========================================

print("\n========== SELECTED EPISODE INSPECTION ==========")

episodes_to_inspect = [1, 2, 3, 9, 11, 14, 15, 17, 18]

for episode_number in episodes_to_inspect:
    if episode_number > len(event_rows):
        print(f"\nEpisode {episode_number}: Not found")
        continue

    event = event_rows[episode_number - 1]
    start = event["start_time"]
    end = event["end_time"]

    mask = (
        (timestamps >= start - CONTEXT_SECONDS)
        & (timestamps <= end + CONTEXT_SECONDS)
    )

    print(f"\n--- EPISODE {episode_number} ---")
    print("Duration:", round(event["duration_seconds"], 2), "seconds")
    print("Minimum MAP:", event["minimum_MAP"])
    print("Mean MAP:", round(event["mean_MAP"], 2))
    print("\nTimestamp | MAP | Valid positive | Below 65")

    for timestamp, value in zip(timestamps[mask], map_values[mask]):
        is_valid = bool(np.isfinite(value) and value > 0)
        is_below = bool(is_valid and value < MAP_THRESHOLD)

        print(
            f"{timestamp:.3f} | {value:.2f} | "
            f"{is_valid} | {is_below}"
        )

print("\nSelected episode inspection completed.")


# ==========================================
# 5. CROSS-SIGNAL VALIDATION OF MAP EPISODES
# ==========================================

def match_signal_to_map(reference_times, signal_times, signal_values,
                        tolerance=MATCH_TOLERANCE):
    """
    Match each MAP reference timestamp to the nearest signal observation
    within the specified tolerance.

    Note: this is nearest-neighbour matching, not one-to-one matching;
    a signal observation can be reused for adjacent reference timestamps.
    """
    matched_values = np.full(len(reference_times), np.nan)

    if len(signal_times) == 0:
        return matched_values

    positions = np.searchsorted(signal_times, reference_times)

    for i, (reference_time, position) in enumerate(
        zip(reference_times, positions)
    ):
        candidates = []

        if position > 0:
            candidates.append(position - 1)

        if position < len(signal_times):
            candidates.append(position)

        if not candidates:
            continue

        nearest = min(
            candidates,
            key=lambda j: abs(signal_times[j] - reference_time),
        )

        difference = abs(signal_times[nearest] - reference_time)

        if difference <= tolerance:
            value = signal_values[nearest]

            if np.isfinite(value) and value > 0:
                matched_values[i] = value

    return matched_values


print("\n========== CROSS-SIGNAL EPISODE VALIDATION ==========")

episodes_to_validate = [15, 17, 18]

for episode_number in episodes_to_validate:
    if episode_number > len(event_rows):
        print(f"\nEpisode {episode_number}: Not found")
        continue

    event = event_rows[episode_number - 1]
    start = event["start_time"]
    end = event["end_time"]

    print(f"\n\n========== EPISODE {episode_number} ==========")
    print("MAP start:", round(start, 3))
    print("MAP end:", round(end, 3))
    print("MAP duration:", round(end - start, 2), "seconds")

    context_mask = (
        (timestamps >= start - CONTEXT_SECONDS)
        & (timestamps <= end + CONTEXT_SECONDS)
    )

    reference_times = timestamps[context_mask]

    periods = {
        "BEFORE": reference_times < start,
        "DURING": (reference_times >= start) & (reference_times <= end),
        "AFTER": reference_times > end,
    }

    for track_name in TRACKS:
        if track_name not in signal_data:
            print(f"\nTRACK: {track_name} - unavailable")
            continue

        signal_times, signal_values = signal_data[track_name]

        matched_values = match_signal_to_map(
            reference_times,
            signal_times,
            signal_values,
            MATCH_TOLERANCE,
        )

        print(f"\nTRACK: {track_name}")

        for period_name, period_mask in periods.items():
            period_values = matched_values[period_mask]
            valid_values = period_values[np.isfinite(period_values)]

            if len(valid_values) == 0:
                print(f"{period_name}: No valid matched observations")
                continue

            print(
                f"{period_name}: "
                f"Count={len(valid_values)}, "
                f"Median={np.median(valid_values):.2f}, "
                f"Mean={np.mean(valid_values):.2f}, "
                f"Min={np.min(valid_values):.2f}, "
                f"Max={np.max(valid_values):.2f}"
            )

print("\nCross-signal episode validation completed.")


# ==========================================
# 6. ARTERIAL PRESSURE CONSISTENCY CHECK
# ==========================================

print("\n========== ARTERIAL PRESSURE CONSISTENCY ==========")

required_pressure_tracks = [
    "Solar8000/ART_MBP",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
]

if all(name in signal_data for name in required_pressure_tracks):
    map_times, map_data = signal_data["Solar8000/ART_MBP"]
    sbp_times, sbp_data = signal_data["Solar8000/ART_SBP"]
    dbp_times, dbp_data = signal_data["Solar8000/ART_DBP"]

    episodes_to_check = [15, 17, 18]

    for episode_number in episodes_to_check:
        if episode_number > len(event_rows):
            print(f"\nEpisode {episode_number}: Not found")
            continue

        event = event_rows[episode_number - 1]
        start = event["start_time"]
        end = event["end_time"]

        mask = (map_times >= start) & (map_times <= end)
        reference_times = map_times[mask]
        observed_map = map_data[mask]

        matched_sbp = match_signal_to_map(
            reference_times, sbp_times, sbp_data, MATCH_TOLERANCE
        )
        matched_dbp = match_signal_to_map(
            reference_times, dbp_times, dbp_data, MATCH_TOLERANCE
        )

        complete = (
            np.isfinite(observed_map)
            & np.isfinite(matched_sbp)
            & np.isfinite(matched_dbp)
            & (observed_map > 0)
            & (matched_sbp > 0)
            & (matched_dbp > 0)
        )

        observed_map = observed_map[complete]
        matched_sbp = matched_sbp[complete]
        matched_dbp = matched_dbp[complete]

        if len(observed_map) == 0:
            print(f"\nEpisode {episode_number}: No complete observations")
            continue

        calculated_map = (matched_sbp + 2 * matched_dbp) / 3
        absolute_difference = np.abs(observed_map - calculated_map)

        print(f"\nEpisode {episode_number}")
        print("Complete observations:", len(observed_map))
        print("Observed MAP median:", round(float(np.median(observed_map)), 2))
        print(
            "Calculated MAP median:",
            round(float(np.median(calculated_map)), 2),
        )
        print(
            "Absolute difference median:",
            round(float(np.median(absolute_difference)), 2),
        )
        print(
            "Absolute difference 95th percentile:",
            round(float(np.percentile(absolute_difference, 95)), 2),
        )
        print(
            "Percentage with difference <= 5 mmHg:",
            round(float(np.mean(absolute_difference <= 5) * 100), 2),
            "%",
        )
else:
    print("Pressure consistency check skipped: one or more pressure tracks are missing.")

print("\nArterial pressure consistency check completed.")


# ==========================================
# 7. EXTREME LOW-MAP EPISODE VALIDATION
# ==========================================

print("\n========== EXTREME LOW-MAP VALIDATION ==========")

episodes_to_check = [1, 2, 3]

for episode_number in episodes_to_check:
    if episode_number > len(event_rows):
        print(f"\nEpisode {episode_number}: Not found")
        continue

    event = event_rows[episode_number - 1]
    start = event["start_time"]
    end = event["end_time"]
    duration = event["duration_seconds"]

    print(f"\nEpisode {episode_number}")
    print("Duration:", round(duration, 2), "seconds")
    print("MAP minimum:", event["minimum_MAP"])
    print("MAP mean:", round(event["mean_MAP"], 2))

    context_start = start - CONTEXT_SECONDS
    context_end = end + CONTEXT_SECONDS

    reference_times = timestamps[
        (timestamps >= context_start) & (timestamps <= context_end)
    ]

    before_mask = reference_times < start
    during_mask = (reference_times >= start) & (reference_times <= end)
    after_mask = reference_times > end

    periods = [
        ("Before", before_mask),
        ("During", during_mask),
        ("After", after_mask),
    ]

    for track_name in TRACKS:
        print(f"\n{track_name}")

        if track_name not in signal_data:
            print("Track unavailable")
            continue

        signal_times, signal_values = signal_data[track_name]

        matched = match_signal_to_map(
            reference_times,
            signal_times,
            signal_values,
            MATCH_TOLERANCE,
        )

        for label, period_mask in periods:
            selected = matched[period_mask]
            selected = selected[np.isfinite(selected) & (selected > 0)]

            if len(selected) == 0:
                print(f"{label}: No valid observations")
                continue

            print(
                f"{label}: count={len(selected)}, "
                f"median={np.median(selected):.2f}, "
                f"min={np.min(selected):.2f}, "
                f"max={np.max(selected):.2f}"
            )

print("\nExtreme low-MAP validation completed.")
# ==========================================
# 8. CHECK AVAILABLE ARTERIAL WAVEFORM TRACKS
# ==========================================

print("\n========== AVAILABLE ARTERIAL TRACKS ==========")

all_tracks = vf.get_track_names()

for track_name in all_tracks:

    if "ART" in track_name.upper():

        print(track_name)

print("\nArterial track inspection completed.")