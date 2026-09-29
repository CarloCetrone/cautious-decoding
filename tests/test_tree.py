# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CautiousTree data structure and pruning logic.
"""

import math
from cautious_decoding.tree import CautiousTree, TreeNode


def test_tree_construction_and_pruning():
    # Breadth = 3, Depth = 3
    breadth = 3
    depth = 3
    tree = CautiousTree(breadth=breadth, depth=depth, eos_token_id=999)

    # Step 1: Expand root to depth 1 (3 nodes)
    frontier = tree.get_leaves()
    assert len(frontier) == 1
    assert frontier[0] is tree.root

    for i in range(breadth):
        tree.root.add_child(token_id=10 + i, logprob=-1.0 - i * 0.1)

    assert tree.current_max_depth() == 1
    frontier = tree.get_leaves()
    assert len(frontier) == 3

    # Step 2: Expand to depth 2 (3 * 3 = 9 nodes)
    for parent in frontier:
        for j in range(breadth):
            parent.add_child(token_id=100 + j, logprob=-0.5 - j * 0.05)

    assert tree.current_max_depth() == 2
    frontier = tree.get_leaves()
    assert len(frontier) == 9

    # Step 3: Expand to depth 3 (9 * 3 = 27 nodes)
    for parent in frontier:
        for k in range(breadth):
            # Give the branch starting with token_id=11 a higher logprob (lower perplexity)
            root_first = parent.get_path_from_root()[0].token_id
            bonus = 0.5 if root_first == 11 else 0.0
            parent.add_child(token_id=1000 + k, logprob=-0.8 + bonus)

    assert tree.current_max_depth() == 3
    paths = tree.get_all_root_to_leaf_paths()
    assert len(paths) == 27

    # Select best path and commit
    best_path, min_ppl = tree.select_best_path()
    assert len(best_path) == 3
    assert best_path[0].token_id == 11  # Received bonus logprob

    # Commit and prune
    committed_token, _, ppl = tree.commit_and_prune()
    assert committed_token == 11
    assert tree.committed_tokens == [11]

    # After pruning, root is now node 11.
    # The remaining leaves must be exactly 3^(3-1) = 9 leaves!
    new_frontier = tree.get_leaves()
    assert len(new_frontier) == 9
    assert tree.current_max_depth() == 2

    # In the next step, those 9 leaves are each expanded by 3 -> 27 leaves again at depth 3
    for parent in new_frontier:
        for m in range(breadth):
            parent.add_child(token_id=2000 + m, logprob=-0.7)

    assert tree.current_max_depth() == 3
    assert len(tree.get_all_root_to_leaf_paths()) == 27
    print("Test passed successfully!")


if __name__ == "__main__":
    test_tree_construction_and_pruning()
