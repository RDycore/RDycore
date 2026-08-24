#!/usr/bin/env python3
"""Fetch USGS Short-Term Network (STN) HWM data for Hurricane Harvey.

The STN Flood Event Viewer public API is at https://stn.wim.usgs.gov/STNServices/.
This script queries the Events endpoint to find Hurricane Harvey, then fetches
all HWMs for that event, saving both raw JSON and a flattened CSV.

Usage: python3 fetch_usgs_hwm.py [outdir]
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.error
import csv

OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
API_BASE = "https://stn.wim.usgs.gov/STNServices"

def fetch_json(endpoint):
    """Fetch JSON from the STN API."""
    url = f"{API_BASE}{endpoint}"
    print(f"Fetching: {url}")
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"ERROR fetching {url}: {e}")
        return None

def main():
    # Step 1: Get event ID for Hurricane Harvey
    print("\n=== Step 1: Finding Hurricane Harvey event ===")
    events = fetch_json("/Events.json")
    if not events:
        print("Failed to fetch events")
        return False

    # Events can be a list or in a 'value' field
    if isinstance(events, list):
        event_list = events
    else:
        event_list = events.get('value', events)
        if isinstance(event_list, dict):
            event_list = [event_list]

    harvey_event = None
    for event in event_list:
        event_name = event.get('event_name', '').lower()
        if 'harvey' in event_name:
            harvey_event = event
            print(f"Found: {event.get('event_name')} (ID: {event.get('event_id')})")
            break

    if not harvey_event:
        print(f"Harvey not found in events. Available events:")
        for event in event_list[:10]:
            print(f"  - {event.get('event_name')} (ID: {event.get('event_id')})")
        return False

    event_id = harvey_event['event_id']

    # Step 2: Fetch all HWMs for this event
    print(f"\n=== Step 2: Fetching HWMs for event {event_id} ===")
    hwms = fetch_json(f"/HWMs.json?EventId={event_id}")
    if not hwms:
        print("Failed to fetch HWMs")
        return False

    if isinstance(hwms, list):
        hwm_list = hwms
    else:
        hwm_list = hwms.get('value', hwms)
        if isinstance(hwm_list, dict):
            hwm_list = [hwm_list]

    print(f"Fetched {len(hwm_list)} records from API")

    # Filter to only records with the Harvey event_id
    hwm_list = [h for h in hwm_list if h.get('event_id') == event_id]
    print(f"Filtered to {len(hwm_list)} HWMs with event_id={event_id}")

    # Step 3: Save raw JSON
    raw_json_file = OUTDIR / "hwm_raw.json"
    with open(raw_json_file, "w") as f:
        json.dump(hwm_list, f, indent=2, default=str)
    print(f"Saved raw JSON to {raw_json_file.name}")

    # Step 4: Flatten to CSV
    if hwm_list:
        # Get all unique keys from all HWM records
        all_keys = set()
        for hwm in hwm_list:
            all_keys.update(hwm.keys())
        fieldnames = sorted(all_keys)

        csv_file = OUTDIR / "hwm_raw.csv"
        with open(csv_file, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(hwm_list)
        print(f"Saved flattened CSV to {csv_file.name} ({len(fieldnames)} columns)")

    # Step 5: Write provenance record
    provenance_file = OUTDIR / "PROVENANCE.txt"
    with open(provenance_file, "w") as f:
        f.write(f"USGS STN HWM Data for Hurricane Harvey\n")
        f.write(f"Event: {harvey_event.get('event_name')}\n")
        f.write(f"Event ID: {event_id}\n")
        f.write(f"Fetched: {datetime.now().isoformat()}\n")
        f.write(f"API Base: {API_BASE}\n")
        f.write(f"Records (Harvey-only): {len(hwm_list)}\n")
        f.write(f"Records (API total): {len(hwm_list)} (filtered from full response)\n")
    print(f"Saved provenance to {provenance_file.name}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
