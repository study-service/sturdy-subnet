# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Syeam Bin Abdullah

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import time
import typing

import bittensor as bt
from eth_account.datastructures import SignedMessage

# Bittensor Miner Template:
import sturdy
from sturdy.algo import naive_algorithm

# import base miner class which takes care of most of the boilerplate
from sturdy.algo_custom import naive_algorithm_optimize
from sturdy.base.miner import BaseMinerNeuron
import json
import os

from sturdy.constants import QUERY_TIMEOUT

LOG_RESPONSE_FILE = "response_times.json"

def append_response_log(log_entry, log_file=LOG_RESPONSE_FILE):
    """Append a log entry as a JSON object to a file (one JSON object per line)."""
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n,")
    except Exception as e:
        bt.logging.error(f"Failed to write response log: {e}")

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the
    forward function with your own logic. You may also want to override the blacklist and priority functions according to your
    needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes
    care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can
    override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing
    requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    async def _init_async(self, config=None) -> None:
        await super()._init_async(config=config)

    async def forward(self, synapse: sturdy.protocol.AllocateAssets) -> sturdy.protocol.AllocateAssets:
        """
        Processes the incoming 'AllocateAssets' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.AllocateAssets): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.AllocateAssets: The synapse object with the 'dummy_output' field set to twice the 'dummy_input'
            value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        start_time = time.time()
        bt.logging.debug("Received request AllocateAssets synapse.")
        # try use default greedy alloaction algorithm to generate allocations
        
        allocate_success = False
        bt.logging.debug("Request pool_data_provider", synapse.pool_data_provider)
        bt.logging.debug("Request total_assets", synapse.assets_and_pools["total_assets"])
        requestTimeout = int(synapse.timeout)
        bt.logging.info(f"Request require timeout:{requestTimeout}")
        try:
            synapse.allocations = await naive_algorithm_optimize(self, synapse)
            allocate_success = True
        except Exception as e:
            bt.logging.error(f"Error allocate: {e}")
            # just return the auto vali generated allocations
            synapse.allocations = synapse.allocations
        end_time = time.time()
        elapsed_time = end_time - start_time
        bt.logging.info(f"Processing time for forward: {elapsed_time:.4f} seconds, allocated {allocate_success}")
        if elapsed_time > requestTimeout:
            bt.logging.error(f"Time allocate higher query timeout elapsed_time:{elapsed_time:.2f}s, timeout:{requestTimeout}s")
        
        # Store log time response to file as JSON
        log_entry = {
            "pool_data_provider": synapse.pool_data_provider,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            "hotkey": getattr(synapse.dendrite, "hotkey", None),
            "elapsed_time": elapsed_time,
            "allocate_success": allocate_success,
            "request_timeout": requestTimeout,
            "num_pools": len(synapse.assets_and_pools),
            "allocations": str(synapse.allocations) if hasattr(synapse, "allocations") else None,
            "error": str(e) if not allocate_success and 'e' in locals() else None
        }
        append_response_log(log_entry)
        return synapse

    async def blacklist(self, synapse: sturdy.protocol.AllocateAssets) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.AllocateAssets): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        bt.logging.info("Checking miner blacklist")

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:  # type: ignore[]
            return True, "Hotkey is not registered"

        requesting_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # type: ignore[]
        stake = self.metagraph.S[requesting_uid].item()

        bt.logging.info(f"Requesting UID: {requesting_uid} | Stake at UID: {stake}")
        if requesting_uid != 40:
            bt.logging.info("Only allow validator uid 32")
            return False, "Requesting UID has no validator permit"
        if stake <= self.config.validator.min_stake:
            bt.logging.info(
                f"Hotkey: {synapse.dendrite.hotkey}: stake below minimum threshold of {self.config.validator.min_stake}"  # type: ignore[]
            )
            return True, "Stake below minimum threshold"

        # validator_permit = self.metagraph.validator_permit[requesting_uid].item()
        # if not validator_permit:
        #     bt.logging.info(f"Requesting UID has no validator permit {requesting_uid}")
        #     return True, "Requesting UID has no validator permit"

        bt.logging.trace(f"Allowing request from UID: {requesting_uid}")
        return False, "Allowed"

    async def priority(self, synapse: sturdy.protocol.AllocateAssets) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.AllocateAssets): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index. # type: ignore[]
        priority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority)  # type: ignore[]
        return priority

    async def save_state(self) -> None:
        """Saves the miner state - currently no state to save for miners."""
        bt.logging.info("Miner state saving skipped - no state to save.")
        pass

    async def load_state(self) -> None:
        """Loads the miner state - currently no state to load for miners."""
        bt.logging.info("Miner state loading skipped - no state to load.")
        pass


async def main() -> None:
    miner = await Miner.create()
    async with miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            await asyncio.sleep(30)  # Add await here


# This is the main function, which runs the miner.
if __name__ == "__main__":
    asyncio.run(main())
