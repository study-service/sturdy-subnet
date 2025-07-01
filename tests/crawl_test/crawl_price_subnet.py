import requests
import json
import time
import os
import concurrent.futures

SUBNET_INFO = "tao_subnets.json"
DATA_DIR = "data"

def fetch_subnet_data(subnet):
    try:
        subnetId = subnet.get("subnet")
        if subnetId is None:
            print("Skipping subnet with no ID")
            return
        if subnetId == 0:
            print("Skipping subnet with ID 0") 
            return

        print("Fetching data for subnet:", subnetId)
        time = 1751341727177 # Current time in milliseconds
        url = f"https://taomarketcap.com/api/subnets/{subnetId}/candle_chart?end={time}&type=1d"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        JSON_FILE = os.path.join(DATA_DIR, f"subnet_{subnetId}_price.json")
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Crawled and saved data for subnet {subnetId} to '{JSON_FILE}'")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for subnet {subnetId}: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for subnet {subnetId}: {e}")
    except IOError as e:
        print(f"Error writing file for subnet {subnetId}: {e}")

try:
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Read subnet info from JSON file
    with open(SUBNET_INFO, "r", encoding="utf-8") as f:
        subnets = json.load(f)

    # Use thread pool to fetch data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_subnet_data, subnets)

except Exception as e:
    print(f"Error in main execution: {e}")
