import requests
import csv
import time

# Replace with the real API URL you discovered
API_URL = "https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/v2Aggregators"

# Optional: Add authentication headers if needed
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    # e.g. "Authorization": "Bearer YOUR_TOKEN"
}

CSV_FILE = "dashboard_data.csv"

# First, fetch the data structure to get column names
resp = requests.get(API_URL, headers=HEADERS)
resp.raise_for_status()
data = resp.json()

# Determine the record list and CSV columns
records = data.get("items", [])  # or adapt to actual structure
if not records:
    print("No data found—check the JSON structure.")
    exit()

# Define CSV headers from keys of the first dict
CSV_HEADERS = list(records[0].keys())

# Initialize CSV file
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
    writer.writeheader()
    writer.writerows(records)

print(f"Saved {len(records)} rows to {CSV_FILE}")
