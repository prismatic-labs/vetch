"""Azure OpenAI SDK provider wrapper.

This module handles patching the Azure OpenAI Python SDK to capture
inference metadata. Since Azure OpenAI uses the same openai SDK
(openai.AzureOpenAI), response formats are identical to standard OpenAI.

The key differences from standard OpenAI:
- Azure AD authentication (api_key or DefaultAzureCredential)
- Azure endpoint format: https://{resource}.openai.azure.com/
- Region inference from Azure resource URLs

Supports:
- Sync completions (client.chat.completions.create)
- Async completions (await client.chat.completions.create)
- Streaming completions (stream=True)
- Async streaming completions (stream=True)

Privacy guarantee: We only read model, usage, and timing metadata.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
from typing import Any

from vetch.providers.openai import patch_openai_client

logger = logging.getLogger(__name__)

# Azure region -> grid region mapping
# Maps Azure region codes to carbon grid zone identifiers
AZURE_REGION_MAP: dict[str, str] = {
    # US regions
    "eastus": "US-EAST",
    "eastus2": "US-EAST",
    "centralus": "US-MIDW",
    "northcentralus": "US-MIDW",
    "southcentralus": "US-SW",
    "westus": "US-WEST",
    "westus2": "US-WEST",
    "westus3": "US-WEST",
    # Europe
    "westeurope": "NL",
    "northeurope": "IE",
    "uksouth": "GB",
    "ukwest": "GB",
    "francecentral": "FR",
    "francesouth": "FR",
    "germanywestcentral": "DE",
    "swedencentral": "SE",
    "switzerlandnorth": "CH",
    "norwayeast": "NO",
    # Asia Pacific
    "eastasia": "HK",
    "southeastasia": "SG",
    "japaneast": "JP-TK",
    "japanwest": "JP-KN",
    "koreacentral": "KR",
    "australiaeast": "AU-NSW",
    "centralindia": "IN-WE",
    # Canada
    "canadacentral": "CA-ON",
    "canadaeast": "CA-QC",
    # Brazil
    "brazilsouth": "BR-S",
}


def infer_region_from_azure_url(base_url: str | None) -> str | None:
    """Infer grid region from Azure OpenAI endpoint URL.

    Supports URL patterns:
    - https://{resource}.openai.azure.com/
    - https://{region}.api.cognitive.microsoft.com/

    Args:
        base_url: The client's base URL.

    Returns:
        Grid region string or None if not determinable.
    """
    if base_url is None:
        return None

    # Pattern 1: region directly in subdomain of cognitive services
    # e.g., https://eastus.api.cognitive.microsoft.com/
    cognitive_match = re.match(
        r"https://([a-z0-9-]+)\.api\.cognitive\.microsoft\.com",
        base_url,
    )
    if cognitive_match:
        azure_region = cognitive_match.group(1)
        return AZURE_REGION_MAP.get(azure_region, azure_region)

    # Pattern 2: resource name in subdomain of openai.azure.com
    # e.g., https://my-resource.openai.azure.com/
    # The region isn't in the URL itself, but we can try to extract
    # from the resource name if it contains a region hint
    azure_match = re.match(
        r"https://([a-z0-9-]+)\.openai\.azure\.com",
        base_url,
    )
    if azure_match:
        resource_name = azure_match.group(1)
        # Check if resource name contains a known region
        for region_code in AZURE_REGION_MAP:
            if region_code in resource_name:
                return AZURE_REGION_MAP[region_code]

    return None


# Track if module is instrumented
_module_instrumented = False

# Store original __init__ methods for uninstrumentation
_original_azure_init: Any = None
_original_async_azure_init: Any = None
_instrument_lock = threading.Lock()


def instrument_azure_openai_module() -> bool:
    """Instrument the Azure OpenAI module to auto-track all client instances.

    Patches AzureOpenAI and AsyncAzureOpenAI __init__ to automatically
    call patch_openai_client on every new client instance. Since Azure
    OpenAI responses use the same format as standard OpenAI, the same
    patching logic applies.

    Returns:
        True if instrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_azure_init, _original_async_azure_init
    import sys

    if _module_instrumented:
        return True

    if "openai" not in sys.modules:
        return False

    try:
        import openai  # type: ignore[import-not-found]

        # Check that AzureOpenAI class exists (openai >= 1.0)
        if not hasattr(openai, "AzureOpenAI"):
            logger.debug("openai.AzureOpenAI not found (requires openai >= 1.0)")
            return False

        with _instrument_lock:
            if _module_instrumented:
                return True

            # Store original __init__ for later restoration
            _original_azure_init = openai.AzureOpenAI.__init__

            def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
                _original_azure_init(self, *args, **kwargs)
                with contextlib.suppress(Exception):
                    patch_openai_client(self)

            openai.AzureOpenAI.__init__ = patched_init

            # Also patch AsyncAzureOpenAI if available
            if hasattr(openai, "AsyncAzureOpenAI"):
                _original_async_azure_init = openai.AsyncAzureOpenAI.__init__

                def patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
                    _original_async_azure_init(self, *args, **kwargs)
                    with contextlib.suppress(Exception):
                        patch_openai_client(self)

                openai.AsyncAzureOpenAI.__init__ = patched_async_init

            _module_instrumented = True

        logger.debug("Azure OpenAI module instrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to instrument Azure OpenAI module: {e}")
        return False


def uninstrument_azure_openai_module() -> bool:
    """Remove Vetch instrumentation from Azure OpenAI module.

    Restores the original __init__ methods.

    Returns:
        True if uninstrumentation succeeded, False otherwise.
    """
    global _module_instrumented, _original_azure_init, _original_async_azure_init
    import sys

    if not _module_instrumented:
        return True

    if "openai" not in sys.modules:
        _module_instrumented = False
        return True

    try:
        import openai

        with _instrument_lock:
            # Restore original __init__
            if _original_azure_init is not None and hasattr(openai, "AzureOpenAI"):
                openai.AzureOpenAI.__init__ = _original_azure_init

            # Restore AsyncAzureOpenAI if we patched it
            if _original_async_azure_init is not None and hasattr(
                openai, "AsyncAzureOpenAI"
            ):
                openai.AsyncAzureOpenAI.__init__ = _original_async_azure_init

            _module_instrumented = False
            _original_azure_init = None
            _original_async_azure_init = None

        logger.debug("Azure OpenAI module uninstrumented")
        return True

    except Exception as e:
        logger.debug(f"Failed to uninstrument Azure OpenAI module: {e}")
        return False
