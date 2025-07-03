import asyncio
import websockets
import json
import csv
import os

CSV_FILE = 'opentensor_data.csv'
WS_ENDPOINT = 'wss://lite.chain.opentensor.ai'

# Optional: define the CSV headers you expect
CSV_HEADERS = ['uid', 'hotkey', 'stake', 'rank', 'trust', 'consensus', 'incentive', 'dividends']

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)


async def save_data_to_csv(data):
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([data.get(h, '') for h in CSV_HEADERS])


async def listen():
    async with websockets.connect(WS_ENDPOINT) as websocket:
        print("Connected to OpenTensor WebSocket")

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Optional: filter/process specific fields
                if isinstance(data, dict) and "neurons" in data:
                    for neuron in data["neurons"]:
                        await save_data_to_csv(neuron)

            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)  # wait and try again


if __name__ == '__main__':
    asyncio.run(listen())
