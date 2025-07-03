
import requests
import json
import asyncio
import time
import os
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://us-central1-stu-dashboard-a0ba2.cloudfunctions.net/v2Aggregators"
OUTPUT_FILE = "v2aggregators_data.json"
OUTPUT_FOLDER = "data"

def fetch_and_save(url: str, outputfile: str = OUTPUT_FILE):
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        print(f"🔄 Fetching data from {url} ...")
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        data = response.json()
        with open(outputfile, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved raw JSON data to '{outputfile}'")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")

async def fetch_data_price_subnet():
    loop = asyncio.get_running_loop()
    subnets = await getAllSubnet()
    print("subnets:", subnets)
    now = int(time.time())
    hour_floor = now - (now % 3600)
    end = hour_floor * 1000

    tasks = []
    # Ensure the output folder exists before starting
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for subnet in subnets:
            subnetId = subnet.get('subnet') if isinstance(subnet, dict) else getattr(subnet, 'subnet', None)
            if subnetId == 0:
                continue
            outputfile = os.path.join(OUTPUT_FOLDER, f"subnet_{subnetId}_price.json")
            url = f"https://taomarketcap.com/api/subnets/{subnetId}/candle_chart?end={end}&type=1d"
            # Schedule the blocking function in the thread pool
            task = loop.run_in_executor(executor, fetch_and_save, url, outputfile)
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks)

async def getAllSubnet():
    loop = asyncio.get_running_loop()
    def fetch():
        response = requests.get("https://taomarketcap.com/api/subnets")
        return response.json()
    return await loop.run_in_executor(None, fetch)

if __name__ == "__main__":
    async def main():
        await fetch_data_price_subnet()
    asyncio.run(main())