import asyncio
import websockets
import json
import csv
import os

WS_URL = "wss://archive.chain.opentensor.ai:443"
CSV_FILE = "archive_opentensor_data.csv"
HEADERS = ['block', 'uid', 'hotkey', 'stake', 'rank', 'trust', 'consensus', 'incentive', 'dividends']

# Initialize CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)

async def save_to_csv(neuron: dict):
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [neuron.get(h, '') for h in HEADERS]
        writer.writerow(row)

async def listen_archive():
    async with websockets.connect(WS_URL) as websocket:
        print("Connected to archive WebSocket")

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Expected data: contains block info and list of neurons
                block = data.get('block')
                neurons = data.get('neurons', [])

                for n in neurons:
                    n['block'] = block  # append block info
                    await save_to_csv(n)
                    print(f"Saved neuron at block {block} - UID: {n.get('uid')}")

            except Exception as e:
                print("Error receiving/parsing data:", e)
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(listen_archive())
