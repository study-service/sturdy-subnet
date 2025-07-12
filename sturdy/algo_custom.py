import math
from typing import Dict, cast

import bittensor as bt
from web3.constants import ADDRESS_ZERO

from sturdy.base.miner import BaseMinerNeuron
from sturdy.pools import (
    POOL_TYPES,
    BittensorAlphaTokenPool,
    ChainBasedPoolModel,
    PoolFactory,
    get_minimum_allocation,
    DaiSavingsRate,
)
from sturdy.protocol import AllocateAssets, AlphaTokenPoolAllocation


import time

from typing import List, Dict, Tuple
from pathlib import Path


THRESHOLD = 0.99  # used to avoid over-allocations


# NOTE: THIS IS JUST AN EXAMPLE - THIS IS NOT VERY OPTIMIZED
async def naive_algorithm_optimize(self: BaseMinerNeuron, synapse: AllocateAssets) -> dict:
    bt.logging.info("Process allocate by naive_algorithm_optimize")
    bt.logging.debug(f"received request type: {synapse.request_type}")
    # bt.logging.debug(f"synapse: {synapse}")
    pools = cast(dict, synapse.assets_and_pools["pools"])

    for uid, pool in pools.items():
        if isinstance(pool, BittensorAlphaTokenPool):
            pools[uid] = PoolFactory.create_pool(
                pool_type=pool.pool_type,
                netuid=pool.netuid,
                current_amount=pool.current_amount,
                pool_data_provider_type=pool.pool_data_provider_type,
            )
        else:
            pools[uid] = PoolFactory.create_pool(
                pool_type=pool.pool_type,
                web3_provider=self.pool_data_providers[pool.pool_data_provider_type],  # type: ignore[]
                user_address=(
                    pool.user_address if pool.user_address != ADDRESS_ZERO else synapse.user_address
                ),
                contract_address=pool.contract_address,
            )
    bt.logging.debug("created pools")

    total_assets_available = int(THRESHOLD * synapse.assets_and_pools["total_assets"])
    pools = cast(dict, synapse.assets_and_pools["pools"])

    if not pools:
        return {}

    rates_sum = 0
    rates = {}
    supply_rate_cache = {}

    # sync pool parameters by calling smart contracts on chain
    for pool in pools.values():
        await pool.sync(self.pool_data_providers[pool.pool_data_provider_type])

    # check the amounts that have been borrowed from the pools - and account for them
    
    minimums = {}
    for pool_uid, pool in pools.items():
        if isinstance(pool, BittensorAlphaTokenPool):
            minimums[pool_uid] = 0
        else:
            minimums[pool_uid] = get_minimum_allocation(pool)
    bt.logging.debug("calculated minimum allocation amounts")

    total_assets_available -= sum(minimums.values())
    balance = int(total_assets_available)
    N = len(pools)
    if N == 0:
        return {}

    # rates are determined by making on chain calls to smart contracts
    # for pool in pools.values():
    #     match pool.pool_type:
    #         case POOL_TYPES.DAI_SAVINGS:
    #             key = (id(pool), None)
    #             if key not in supply_rate_cache:
    #                 supply_rate_cache[key] = await pool.supply_rate()
    #             apy = supply_rate_cache[key]
    #             rates[pool.contract_address] = apy
    #             rates_sum += apy
    #         case POOL_TYPES.BT_ALPHA:
    #             price = pool._price_rao
    #             rates[str(pool.netuid)] = price
    #             rates_sum += price
    #         case _:
    #             amount = balance // N
    #             key = (id(pool), amount)
    #             if key not in supply_rate_cache:
    #                 supply_rate_cache[key] = await pool.supply_rate(amount)
    #             apy = supply_rate_cache[key]
    #             rates[pool.contract_address] = apy
    #             rates_sum += apy

    # check the type of the first pool, if it's a bittensor alpha token pool then assume the rest are too
    first_pool = next(iter(pools.values()))
    delegate_ss58 = "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3"  # This is OTF's hotkey
    # by default we just distribute tao equally lol
    bt.logging.debug(f"Number pools: {len(pools)}")
    if first_pool.pool_type == POOL_TYPES.BT_ALPHA:
        self.pool_data_providers[first_pool.pool_data_provider_type]
        allocation_dict = {
            netuid: AlphaTokenPoolAllocation(delegate_ss58=delegate_ss58, amount=math.floor(balance / N)) for netuid in pools
        }
        total_allocated = sum(
            alloc.amount if isinstance(alloc, AlphaTokenPoolAllocation) else alloc
            for alloc in allocation_dict.values()
        )
        bt.logging.info(f"Total allocated: {total_allocated}")
        # Log percentage allocated in each pool
        for netuid, alloc in allocation_dict.items():
            percent = (alloc.amount / total_allocated * 100) if total_allocated > 0 else 0
            bt.logging.info(f"Pool {netuid}: {percent:.2f}% allocated")
        return allocation_dict
    
    
    # --- Add minimum allocation to result ---
    pool_list = list(pools.values())
    pool_uids = list(pools.keys())
    minimums_list = [minimums[uid] for uid in pool_uids]
    result = await optimize_allocation(pool_list, balance, step = max(10**16, balance // 100))
    
    # Add minimums to each allocation
    final_allocations = [alloc + min_alloc for alloc, min_alloc in zip(result, minimums_list)]
    # Return as a dict mapping pool_uid to allocation
    allocation_dict = {uid: alloc for uid, alloc in zip(pool_uids, final_allocations)}
    bt.logging.debug(f"result optimize_allocation: {result}")
    bt.logging.debug(f"minimum allocate {sum(minimums.values())}")
    bt.logging.debug(f"allocation_dict: {allocation_dict}")
    bt.logging.debug(f"Balance to allocate {balance}")
    bt.logging.info(f"Total allocated: {sum(final_allocations)}")
    bt.logging.info(f"Remain % balance {(balance + sum(minimums.values()) - sum(final_allocations)) / balance}")
    # Log percentage allocated in each pool
    total_allocated = sum(final_allocations)
    for uid, alloc in allocation_dict.items():
        percent = (alloc / total_allocated * 100) if total_allocated > 0 else 0
        bt.logging.info(f"Pool {uid}: {percent:.2f}% allocated, allocate: {alloc}")
    return allocation_dict

class SegmentTree:
    """
    A segment tree for efficiently finding and updating the maximum marginal APY and its index.
    This is similar to a max-heap but allows for efficient range queries and updates.
    """

    def __init__(self, n: int):
        """
        Initialize the segment tree with n leaves.

        Args:
            n (int): Number of pools (leaves in the segment tree).
        """
        self.n = n
        # Compute the size as the next power of two >= n
        self.size = 1 << ((n - 1).bit_length())
        # Each node stores a tuple: (marginal_apy, index)
        self.data = [(-math.inf, -1) for _ in range(2 * self.size)]

    def build(self, marginals: List[float]):
        """
        Build the segment tree from a list of marginal APYs.

        Args:
            marginals (list[float]): List of marginal APYs for each pool.
        """
        for i, marginal in enumerate(marginals):
            self.data[self.size + i] = (marginal, i)

        for i in reversed(range(1, self.size)):
            left = self.data[2 * i]
            right = self.data[2 * i + 1]
            self.data[i] = left if left[0] >= right[0] else right

    def update(self, idx: int, value: float):
        """
        Update the marginal APY for a specific pool and propagate the change up the tree.

        Args:
            idx (int): Index of the pool to update.
            value (float): New marginal APY value.
        """
        pos = self.size + idx
        self.data[pos] = (value, idx)
        pos //= 2
        while pos >= 1:
            left = self.data[2 * pos]
            right = self.data[2 * pos + 1]
            self.data[pos] = left if left[0] >= right[0] else right
            pos //= 2

    def query(self) -> Tuple[float, int]:
        """
        Query the maximum marginal APY and its index.

        Returns:
            tuple: (max_marginal_apy, index)
        """
        return self.data[1]




DECIMALS = 10**18
DEFAULT_STEP = 10**15

def round_amount(amount: int, step: int = DEFAULT_STEP) -> int:
    return (amount // step) * step

async def get_cached_rate(
    pool: ChainBasedPoolModel,
    amount: int,
    cache: Dict[Tuple[str, int], float],
    step: int
) -> float:
    safe_amount = max(amount, step)  # tránh amount=0
    rounded = round_amount(safe_amount, step)
    key = (id(pool), rounded)
    if key not in cache:
        start_time = time.time()
        # bt.logging.debug(f"[Cache Miss] supply_rate key: {key}")
        try:
            if isinstance(pool, DaiSavingsRate):
                rate = await pool.supply_rate()
            else:
                rate = await pool.supply_rate(rounded)
            cache[key] = rate
        except Exception as e:
            bt.logging.error(f"Failed to get supply_rate for {key}: {e}")
            cache[key] = 0

        elapsed = (time.time() - start_time) * 1000
        # bt.logging.info(
        #     f"Supply rate | Pool: {key[0]} | Amount: {rounded} | Rate: {cache[key]/DECIMALS:.6f} | Time: {elapsed:.0f}ms"
        # )

    return cache[key]

async def is_fixed_rate(
    pool: ChainBasedPoolModel,
    step: int,
    cache: Dict[Tuple[int, int], float]
) -> bool:
    try:
        if isinstance(pool, DaiSavingsRate):
            return True

        safe_amount = max(step, 1)  # tránh 0
        rate_1x = await get_cached_rate(pool, safe_amount, cache, step)
        rate_2x = await get_cached_rate(pool, safe_amount * 2, cache, step)

        return math.isclose(rate_1x, rate_2x, rel_tol=1e-9)
    except Exception as e:
        bt.logging.error(f"Failed in is_fixed_rate: {e}")
        return False

async def optimize_allocation(
    pools: List[ChainBasedPoolModel],
    total_amount: int,
    step: int = 10**15
) -> List[int]:
    if step <= 0 or total_amount <= 0 or not pools:
        return [0] * len(pools)

    n = len(pools)
    allocations = [0] * n
    remaining = total_amount
    supply_rate_cache: Dict[Tuple[int, int], float] = {}

    # Handle fixed-rate
    fixed_rate_pools = []
    fixed_rates = []
    for i, pool in enumerate(pools):
        if await is_fixed_rate(pool, step, supply_rate_cache):
            apy = await get_cached_rate(pool, step, supply_rate_cache, step)
            fixed_rate_pools.append(i)
            fixed_rates.append(apy)

    if len(fixed_rate_pools) == n:
        best_idx = fixed_rate_pools[fixed_rates.index(max(fixed_rates))]
        allocations[best_idx] = total_amount
        return allocations

    # Initialize marginal APYs
    marginals = []
    for i, pool in enumerate(pools):
        try:
            apy_now = await get_cached_rate(pool, step, supply_rate_cache, step)
            apy_next = await get_cached_rate(pool, step * 2, supply_rate_cache, step)
            margin = max(apy_next - apy_now, 1e-9)
            marginals.append(margin)
        except Exception:
            marginals.append(-math.inf)

    seg = SegmentTree(n)
    seg.build(marginals)

    # Greedy allocation loop
    while remaining > 0:
        best_marginal_apy, best_idx = seg.query()
        if best_marginal_apy <= 0 or best_idx == -1:
            break

        actual_step = min(step, remaining)
        allocations[best_idx] += actual_step
        remaining -= actual_step

        try:
            current = allocations[best_idx]
            apy_now = await get_cached_rate(pools[best_idx], current, supply_rate_cache, step)
            apy_next = await get_cached_rate(pools[best_idx], current + step, supply_rate_cache, step)
            new_marginal = max(apy_next - apy_now, 1e-9)
        except Exception:
            new_marginal = -math.inf

        seg.update(best_idx, new_marginal)

    return allocations


