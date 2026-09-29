# SPDX-License-Identifier: Apache-2.0
"""
Example inference script demonstrating Cautious Tree Search Decoding with vLLM.
"""

from vllm import LLM
from cautious_decoding import CautiousDecoder


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading vLLM with model {model_name} and Prefix Caching enabled...")
    llm = LLM(
        model=model_name,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
    )

    prompt = "Solve step-by-step: If a train travels at 60 mph for 2.5 hours, how far does it travel?"
    print(f"\nPrompt: {prompt}\n")

    # Run with breadth=3 and depth=3
    # 27 candidate paths evaluated -> commit 1 token -> prune to 9 -> expand back to 27
    decoder = CautiousDecoder(
        llm=llm,
        breadth=3,
        depth=3,
        temperature=0.7,
        max_tokens=128,
    )

    result = decoder.generate(prompt=prompt, verbose=True)

    print("\n" + "=" * 50)
    print("FINAL GENERATED TEXT:")
    print("=" * 50)
    print(result["generated_text"])
    print("=" * 50)
    print(f"Committed Tokens: {result['num_committed_tokens']}")
    print(f"Elapsed Time: {result['elapsed_time_sec']:.2f}s")
    print(f"Speed: {result['tokens_per_second']:.2f} tokens/s")
    print(f"Stats: {result['stats']}")


if __name__ == "__main__":
    main()
