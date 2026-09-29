# SPDX-License-Identifier: Apache-2.0
"""
Cautious Tree Search Decoder implementation for vLLM.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from cautious_decoding.tree import CautiousTree, TreeNode


class CautiousDecoder:
    """Executes Cautious Tree Search Decoding (CTSD) with prefix-cached vLLM execution.

    Parameters:
        llm: An instance of `vllm.LLM`
        breadth: Branching factor B (number of candidate tokens sampled at each frontier node).
        depth: Maximum lookahead tree depth D before pruning.
        temperature: Sampling temperature for logits.
        max_tokens: Maximum number of tokens to commit.
    """

    def __init__(
        self,
        llm: Any,
        breadth: int = 3,
        depth: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Extract tokenizer and EOS token from vLLM model
        self.tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def generate(
        self,
        prompt: str,
        breadth: Optional[int] = None,
        depth: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Runs Cautious Tree Search Decoding on the given prompt.

        Returns a dictionary containing:
            - prompt: Original input prompt.
            - generated_text: Generated decoded text.
            - output_tokens: List of committed token IDs.
            - num_committed_tokens: Count of committed tokens.
            - elapsed_time: Total wall-clock time in seconds.
            - stats: Execution metrics (evaluations, forward passes).
        """
        import vllm

        breadth = breadth or self.breadth
        depth = depth or self.depth
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        start_time = time.time()

        # Tokenize initial prompt
        prompt_token_ids: List[int] = self.tokenizer.encode(prompt)

        # Initialize the search tree
        tree = CautiousTree(
            breadth=breadth,
            depth=depth,
            eos_token_id=self.eos_token_id,
        )

        num_forward_passes = 0
        num_prunings = 0

        # Step 1: Initial Rollout to build the tree up to `depth`
        # Depth 1: B sequences
        # Depth 2: B^2 sequences
        # ...
        # Depth D: B^D sequences (e.g., 3^3 = 27)
        while len(tree.committed_tokens) < max_tokens:
            frontier_leaves = tree.get_leaves()

            # If tree has reached target depth D, evaluate and prune to B^(D-1)
            if tree.current_max_depth() >= depth:
                committed_token, _, ppl = tree.commit_and_prune()
                num_prunings += 1

                if verbose:
                    token_str = self.tokenizer.decode([committed_token])
                    print(
                        f"[CTSD Commit] Token: {token_str!r} (ID: {committed_token}) | "
                        f"PPL: {ppl:.2f} | Tree leaves remaining: {len(tree.get_leaves())}"
                    )

                if committed_token == self.eos_token_id:
                    break

                if len(tree.committed_tokens) >= max_tokens:
                    break

                # After pruning, the tree has depth D-1 with B^(D-1) remaining leaves
                # The next iteration will expand these B^(D-1) leaves back to depth D (B^D leaves)
                frontier_leaves = tree.get_leaves()

            # Prepare batch of sequences for all frontier leaves
            batch_token_ids: List[List[int]] = []
            for leaf in frontier_leaves:
                seq = prompt_token_ids + tree.committed_tokens
                if leaf is not tree.root:
                    seq = seq + leaf.get_token_ids_from_root()
                batch_token_ids.append(seq)

            # Build SamplingParams to get top-B token logprobs for each leaf
            sampling_params = vllm.SamplingParams(
                max_tokens=1,
                temperature=max(temperature, 1e-5),
                logprobs=breadth,
            )

            # Batched forward pass over all active leaves (reusing prefix KV cache)
            outputs = self.llm.generate(
                prompt_token_ids=batch_token_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            num_forward_passes += len(batch_token_ids)

            # Attach top-B candidate children to each parent leaf
            for leaf, out in zip(frontier_leaves, outputs):
                if not out.outputs:
                    continue
                first_out = out.outputs[0]
                logprobs_dict = first_out.logprobs[0] if first_out.logprobs else {}

                # Sort top tokens by logprob descending and select top B
                sorted_tokens = sorted(
                    logprobs_dict.items(),
                    key=lambda item: item[1].logprob if hasattr(item[1], "logprob") else item[1],
                    reverse=True,
                )[:breadth]

                # Fallback to sampled token if logprobs_dict has fewer than breadth
                if not sorted_tokens:
                    tok_id = first_out.token_id
                    sorted_tokens = [(tok_id, 0.0)]

                for tok_id, logprob_obj in sorted_tokens:
                    lp = logprob_obj.logprob if hasattr(logprob_obj, "logprob") else float(logprob_obj)
                    tok_text = getattr(logprob_obj, "decoded_token", None)
                    leaf.add_child(token_id=tok_id, logprob=lp, token_text=tok_text)

        # Drain any remaining tokens along the best path if finished before EOS
        if tree.get_all_root_to_leaf_paths() and (
            not tree.committed_tokens or tree.committed_tokens[-1] != self.eos_token_id
        ):
            best_path, _ = tree.select_best_path()
            for node in best_path:
                if len(tree.committed_tokens) >= max_tokens:
                    break
                tree.committed_tokens.append(node.token_id)
                if node.token_id == self.eos_token_id:
                    break

        elapsed = time.time() - start_time
        generated_text = self.tokenizer.decode(tree.committed_tokens)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "output_tokens": tree.committed_tokens,
            "num_committed_tokens": len(tree.committed_tokens),
            "elapsed_time_sec": elapsed,
            "tokens_per_second": len(tree.committed_tokens) / max(elapsed, 1e-4),
            "stats": {
                "breadth": breadth,
                "depth": depth,
                "num_forward_passes": num_forward_passes,
                "num_prunings": num_prunings,
            },
        }
