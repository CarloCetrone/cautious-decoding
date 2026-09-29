# SPDX-License-Identifier: Apache-2.0
"""
vLLM plugin entrypoint for Cautious Tree Search Decoding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def register():
    """Plugin initialization function called by vLLM's plugin discovery system."""
    try:
        import vllm
        from cautious_decoding.decoder import CautiousDecoder

        # Monkey-patch or attach helper method onto vllm.LLM
        def cautious_generate(
            self,
            prompt: str,
            breadth: int = 3,
            depth: int = 3,
            max_tokens: int = 512,
            temperature: float = 0.7,
            verbose: bool = False,
        ) -> Dict[str, Any]:
            """Performs Cautious Tree Search Decoding on the prompt."""
            decoder = CautiousDecoder(
                llm=self,
                breadth=breadth,
                depth=depth,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return decoder.generate(prompt=prompt, verbose=verbose)

        vllm.LLM.cautious_generate = cautious_generate

        logger.info("[CautiousDecoding] vLLM plugin registered successfully: `LLM.cautious_generate` is available.")
    except Exception as e:
        logger.warning(f"[CautiousDecoding] Failed to attach plugin to vllm.LLM: {e}")
