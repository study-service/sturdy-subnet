import asyncio
from typing import Any, Dict
import json
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sturdy.pool_registry.pool_registry import POOL_REGISTRY
from sturdy.pools import PoolFactory, POOL_TYPES


async def crawl_pool_data(synapse: Any, test_amount: int = 1000) -> Dict[str, Dict[str, Any]]:
    """
    Crawl APY (supply_rate) and current allocated volume for all pools in the synapse.
    Returns a dict: pool_uid -> {'apy': ..., 'allocated_volume': ...}
    """
    pools = synapse.assets_and_pools["pools"]
    results = {}

    for pool_uid, pool in pools.items():
        pool_data = {}

        # --- Get APY ---
        try:
            # Some pools (like DAI Savings) may not require an amount
            if hasattr(pool, "supply_rate"):
                if getattr(pool, "pool_type", None) == "DAI_SAVINGS":
                    apy = await pool.supply_rate()
                else:
                    apy = await pool.supply_rate(test_amount)
                pool_data['apy'] = apy
            else:
                pool_data['apy'] = None
        except Exception as e:
            pool_data['apy'] = f"Error: {e}"

        # --- Get Allocated Volume ---
        try:
            # For most pools, use _user_deposits or current_amount
            if hasattr(pool, "_user_deposits"):
                pool_data['allocated_volume'] = getattr(pool, "_user_deposits")
            elif hasattr(pool, "current_amount"):
                pool_data['allocated_volume'] = getattr(pool, "current_amount")
            else:
                pool_data['allocated_volume'] = None
        except Exception as e:
            pool_data['allocated_volume'] = f"Error: {e}"

        results[pool_uid] = pool_data

    return results

async def crawl_pools_data(pools: list[Any], test_amount: int = 1000) -> dict:
    """
    For a list of pool instances, fetch APY and allocated volume for each pool.
    Returns a dict: pool_uid -> {'apy': ..., 'allocated_volume': ...}
    """
    results = {}
    for pool in pools:
        pool_data = {}
        # Determine pool_uid
        pool_uid = getattr(pool, 'contract_address', None) or getattr(pool, 'netuid', None) or str(id(pool))
        # Get APY
        try:
            if hasattr(pool, 'supply_rate'):
                if getattr(pool, 'pool_type', None) == 'DAI_SAVINGS':
                    apy = await pool.supply_rate()
                else:
                    apy = await pool.supply_rate(test_amount)
                pool_data['apy'] = apy
            else:
                pool_data['apy'] = None
        except Exception as e:
            pool_data['apy'] = f"Error: {e}"
        # Get Allocated Volume
        try:
            if hasattr(pool, '_user_deposits'):
                pool_data['allocated_volume'] = getattr(pool, '_user_deposits')
            elif hasattr(pool, 'current_amount'):
                pool_data['allocated_volume'] = getattr(pool, 'current_amount')
            else:
                pool_data['allocated_volume'] = None
        except Exception as e:
            pool_data['allocated_volume'] = f"Error: {e}"
        results[pool_uid] = pool_data
    return results

async def save_crawled_data_to_json(synapse: Any, filename: str, test_amount: int = 1000):
    """
    Crawl pool data and save the result to a JSON file.
    """
    data = await crawl_pool_data(synapse, test_amount=test_amount)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

async def crawl_all_pool_info(pools: list[Any], test_amount: int = 1000) -> dict:
    """
    Deep crawl: For a list of pool instances, collect all public (non-callable, non-private) attributes,
    as well as APY and allocated volume, and return a dict mapping pool_uid to all info.
    """
    results = {}
    for pool in pools:
        pool_info = {}
        # Pool UID
        pool_uid = getattr(pool, 'contract_address', None) or getattr(pool, 'netuid', None) or str(id(pool))
        # All public attributes (not callable, not private)
        for attr in dir(pool):
            if attr.startswith('_'):
                continue
            try:
                value = getattr(pool, attr)
                if callable(value):
                    continue
                pool_info[attr] = value
            except Exception:
                pool_info[attr] = 'Error reading attribute'
        # APY
        try:
            if hasattr(pool, 'supply_rate'):
                if getattr(pool, 'pool_type', None) == 'DAI_SAVINGS':
                    apy = await pool.supply_rate()
                else:
                    apy = await pool.supply_rate(test_amount)
                pool_info['apy'] = apy
            else:
                pool_info['apy'] = None
        except Exception as e:
            pool_info['apy'] = f"Error: {e}"
        # Allocated volume
        try:
            if hasattr(pool, '_user_deposits'):
                pool_info['allocated_volume'] = getattr(pool, '_user_deposits')
            elif hasattr(pool, 'current_amount'):
                pool_info['allocated_volume'] = getattr(pool, 'current_amount')
            else:
                pool_info['allocated_volume'] = None
        except Exception as e:
            pool_info['allocated_volume'] = f"Error: {e}"
        results[pool_uid] = pool_info
    return results

def convert_to_serializable(obj):
    """Convert complex objects to JSON serializable format"""
    if hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, type):
        return obj.__name__
    elif callable(obj):
        return f"<function {obj.__name__}>"
    else:
        return str(obj)

async def crawl_all_pool_info_to_json(pools: list[Any], filename: str, test_amount: int = 1000):
    """
    Crawl pool data and save the result to a JSON file.
    """
    data = await crawl_all_pool_info(pools, test_amount=test_amount)
    
    # Convert non-serializable objects to strings
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return convert_to_serializable(obj)
    
    serializable_data = make_serializable(data)
    
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)

async def crawl_registry_pools_to_json(filename="all_registry_pools_info.json"):
    all_pools = []
    for entry in POOL_REGISTRY.values():
        pools_dict = entry["assets_and_pools"]["pools"]
        user_address = entry.get("user_address", None)
        for pool_info in pools_dict.values():
            pool_type = POOL_TYPES[pool_info["pool_type"]]
            kwargs = {"contract_address": pool_info["contract_address"]}
            if user_address:
                kwargs["user_address"] = user_address
            pool = PoolFactory.create_pool(pool_type, **kwargs)
            all_pools.append(pool)
    await crawl_all_pool_info_to_json(all_pools, filename)


# Standalone test block (with mock data)
if __name__ == "__main__":
    class MockPool:
        def __init__(self, pool_type, apy, allocated):
            self.pool_type = pool_type
            self._user_deposits = allocated if pool_type != "BT_ALPHA" else None
            self.current_amount = allocated if pool_type == "BT_ALPHA" else None
            self._apy = apy
        async def supply_rate(self, amount=None):
            return self._apy

    class MockSynapse:
        def __init__(self):
            self.assets_and_pools = {
                "pools": {
                    "Aave": MockPool("AAVE_DEFAULT", 0.051, 12000),
                    "Compound": MockPool("COMPOUND_V3", 0.048, 8000),
                    "Morpho": MockPool("MORPHO", 0.053, 5000),
                    "101": MockPool("BT_ALPHA", None, 3000),
                }
            }

    async def main():
        await crawl_registry_pools_to_json('deep_crawled_pool_data.json');
        

    asyncio.run(main()) 