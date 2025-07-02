import math
from typing import cast

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

import json
from typing import List, Tuple
THRESHOLD = 0.99  # used to avoid over-allocations


# NOTE: THIS IS JUST AN EXAMPLE - THIS IS NOT VERY OPTIMIZED
async def naive_algorithm_optimize(self: BaseMinerNeuron, synapse: AllocateAssets) -> dict:
    bt.logging.debug(f"received request type: {synapse.request_type}")
    bt.logging.debug(f"synapse: {synapse}")
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
    bt.logging.debug("synced pools")

    # check the amounts that have been borrowed from the pools - and account for them
    minimums = {}
    for pool_uid, pool in pools.items():
        if isinstance(pool, BittensorAlphaTokenPool):
            minimums[pool_uid] = 0
        else:
            minimums[pool_uid] = get_minimum_allocation(pool)
    bt.logging.debug("set minimum allocation amounts")

    total_assets_available -= sum(minimums.values())
    balance = int(total_assets_available)
    N = len(pools)
    if N == 0:
        return {}

    # rates are determined by making on chain calls to smart contracts
    for pool in pools.values():
        match pool.pool_type:
            case POOL_TYPES.DAI_SAVINGS:
                key = (id(pool), None)
                if key not in supply_rate_cache:
                    supply_rate_cache[key] = await pool.supply_rate()
                apy = supply_rate_cache[key]
                rates[pool.contract_address] = apy
                rates_sum += apy
            case POOL_TYPES.BT_ALPHA:
                price = pool._price_rao
                rates[str(pool.netuid)] = price
                rates_sum += price
            case _:
                amount = balance // N
                key = (id(pool), amount)
                if key not in supply_rate_cache:
                    supply_rate_cache[key] = await pool.supply_rate(amount)
                apy = supply_rate_cache[key]
                rates[pool.contract_address] = apy
                rates_sum += apy

    # check the type of the first pool, if it's a bittensor alpha token pool then assume the rest are too
    first_pool = next(iter(pools.values()))
    delegate_ss58 = "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3"  # This is OTF's hotkey
    # by default we just distribute tao equally lol
    if first_pool.pool_type == POOL_TYPES.BT_ALPHA:
        self.pool_data_providers[first_pool.pool_data_provider_type]
        return {
            netuid: AlphaTokenPoolAllocation(delegate_ss58=delegate_ss58, amount=math.floor(balance / N)) for netuid in pools
        }
    bt.logging.debug(f"pools: {pools}")
    
    result = await optimize_allocation(list(pools.values()), balance, 1_00_000)
    bt.logging.debug(f"result: {result}")
    return result

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


async def optimize_allocation(pools: List[ChainBasedPoolModel], total_amount: int, step: int = 10**18) -> List[int]:
    """
    Allocate total_amount into multiple pools to maximize returns based on marginal APY.
    """
    if step <= 0:
        raise ValueError("Step must be greater than 0")
    if total_amount <= 0 or not pools:
        return [0] * len(pools)

    n = len(pools)
    allocations = [0] * n
    remaining = total_amount

    # --- Detect fixed-rate pools ---
    fixed_rate_pools = []
    fixed_rates = []
    variable_rate_pools = []
    supply_rate_cache = {}
    for i, pool in enumerate(pools):
        key0 = (id(pool), 0)
        key_step = (id(pool), step)
        if await is_fixed_rate(pool, step):
            if isinstance(pool, DaiSavingsRate):
                if key0 not in supply_rate_cache:
                    supply_rate_cache[key0] = await pool.supply_rate()
                apy = supply_rate_cache[key0]
            else:
                if key0 not in supply_rate_cache:
                    supply_rate_cache[key0] = await pool.supply_rate(0)
                apy = supply_rate_cache[key0]
            fixed_rate_pools.append(i)
            fixed_rates.append(apy)
        else:
            variable_rate_pools.append(i)

    # --- If any fixed-rate pool exists, allocate all to the best one ---
    if fixed_rate_pools:
        best_idx = fixed_rate_pools[fixed_rates.index(max(fixed_rates))]
        allocations = [0] * n
        allocations[best_idx] = total_amount
        return allocations

    # --- Greedy allocation for variable-rate pools ---
    # Precompute initial marginal APYs
    marginals = []
    for i, pool in enumerate(pools):
        try:
            key0 = (id(pool), 0)
            key_step = (id(pool), step)
            if isinstance(pool, DaiSavingsRate):
                if key0 not in supply_rate_cache:
                    supply_rate_cache[key0] = await pool.supply_rate()
                apy_now = apy_next = supply_rate_cache[key0]
            else:
                if key0 not in supply_rate_cache:
                    supply_rate_cache[key0] = await pool.supply_rate(0)
                if key_step not in supply_rate_cache:
                    supply_rate_cache[key_step] = await pool.supply_rate(step)
                apy_now = supply_rate_cache[key0]
                apy_next = supply_rate_cache[key_step]
            marginal = apy_next - apy_now
        except Exception:
            marginal = -math.inf
        marginals.append(marginal)

    seg = SegmentTree(n)
    seg.build(marginals)

    while remaining > 0:
        best_marginal_apy, best_idx = seg.query()
        if best_marginal_apy <= 0 or best_idx == -1:
            break

        actual_step = min(step, remaining)
        allocations[best_idx] += actual_step
        remaining -= actual_step

        try:
            pool = pools[best_idx]
            key_now = (id(pool), allocations[best_idx])
            key_next = (id(pool), allocations[best_idx] + step)
            if isinstance(pool, DaiSavingsRate):
                if key_now not in supply_rate_cache:
                    supply_rate_cache[key_now] = await pool.supply_rate()
                apy_now = apy_next = supply_rate_cache[key_now]
            else:
                if key_now not in supply_rate_cache:
                    supply_rate_cache[key_now] = await pool.supply_rate(allocations[best_idx])
                if key_next not in supply_rate_cache:
                    supply_rate_cache[key_next] = await pool.supply_rate(allocations[best_idx] + step)
                apy_now = supply_rate_cache[key_now]
                apy_next = supply_rate_cache[key_next]
            new_marginal = apy_next - apy_now
        except Exception:
            new_marginal = -math.inf

        seg.update(best_idx, new_marginal)

    return allocations

async def is_fixed_rate(pool, step: int) -> bool:
    """Check if a pool has a fixed supply rate regardless of amount."""
    try:
        if isinstance(pool, DaiSavingsRate):
            apy0 = apy1 = await pool.supply_rate()
        else:
            apy0 = await pool.supply_rate(0)
            apy1 = await pool.supply_rate(step)
        return apy0 is not None and apy0 == apy1
    except Exception:
        return False

async def get_apy(pool, amount=None):
    """Get APY for a pool, handling DaiSavingsRate signature."""
    if isinstance(pool, DaiSavingsRate):
        return await pool.supply_rate()
    else:
        return await pool.supply_rate(amount)
