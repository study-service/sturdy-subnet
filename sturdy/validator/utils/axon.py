import traceback

import bittensor as bt
from bittensor import dendrite
from aiohttp.client_exceptions import InvalidUrlClientError

from sturdy.constants import (
    QUERY_TIMEOUT,
)
from sturdy.validator.request import Request


async def query_single_axon(dendrite: dendrite, request: Request, query_timeout: int = QUERY_TIMEOUT) -> Request | None:
    """
    Query a single axon with a request.

    Args:
        dendrite (bt.dendrite): The dendrite to use for querying.
        request (Request): The request to send.

    Returns:
        Request | None: The request with results populated, or None if the request failed.
    """

    try:
        bt.logging.debug(f"🌐 NETWORK CALL: Sending to UID {request.uid} at {request.axon.ip}:{request.axon.port}")
        bt.logging.debug(f"🌐 Timeout: {query_timeout}s")
        
        result = await dendrite.call(
            target_axon=request.axon,
            synapse=request.synapse,
            timeout=query_timeout,
            deserialize=False,
        )

        if not result:
            bt.logging.warning(f"⚠️  NO RESPONSE from UID {request.uid}")
            return None
            
        request.synapse = result
        request.response_time = result.dendrite.process_time if result.dendrite.process_time is not None else query_timeout
        bt.logging.debug(f"query_single_axon response {result}")
        
        request.deserialized = result.deserialize()
        bt.logging.debug(f"Request after set deserialized {request}")
        bt.logging.debug(f"✅ SUCCESS: UID {request.uid} responded in {request.response_time:.2f}s")
        return request

    except InvalidUrlClientError:
        bt.logging.error(f"🚫 INVALID URL: UID {request.uid} at {request.axon.ip}:{request.axon.port}")
        return None

    except Exception as e:
        bt.logging.error(f"💥 NETWORK ERROR: UID {request.uid} - {e}")
        traceback.print_exc()
        return None
