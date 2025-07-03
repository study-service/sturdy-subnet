import requests
import json

API_URL = "https://taomarketcap.com/api/subnets"
JSON_FILE = "tao_subnets.json"

# Step 1: Fetch data from API
response = requests.get(API_URL)
response.raise_for_status()  # Raise error if request failed
data = response.json()

# Step 2: Check format
if not isinstance(data, list) or len(data) == 0:
    print("No data returned or unexpected structure.")
    exit()

# Step 3: Write to JSON file
with open(JSON_FILE, mode="w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"✅ Crawled and saved {len(data)} subnets to '{JSON_FILE}'")
