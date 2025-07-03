import requests
import json
from datetime import datetime, timezone

# URL of the Sturdy Subnet API
url = "https://www.sturdysubnet.org/api/aggregatorHistoricalData"

# Output file path
output_file = "sturdy_data.json"

def fetch_and_store_data():
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses

        data = response.json()

        # Convert timestamp to datetime in UTC for each entry
        for entry in data["data"]:
            timestamp = entry["timestamp"]
            entry["time"] = datetime.fromtimestamp(timestamp/1000, tz=timezone.utc).isoformat()

        # Save data to JSON file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Data saved to {output_file}. Total entries: {len(data['data'])}")
    except requests.RequestException as e:
        print(f"❌ Error fetching data: {e}")

if __name__ == "__main__":
    fetch_and_store_data()
