# SPDX-License-Identifier: Apache-2.0
"""
Tree data structure and pruning algorithms for Cautious Tree Search Decoding (CTSD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TreeNode:
    """A node in the Cautious Tree Search exploration tree."""
    token_id: int
    logprob: float
    parent: Optional[TreeNode] = None
    children: List[TreeNode] = field(default_factory=list)
    cumulative_logprob: float = 0.0
    depth: int = 0
    token_text: Optional[str] = None

    def __post_init__(self):
        if self.parent is not None:
            self.cumulative_logprob = self.parent.cumulative_logprob + self.logprob
            self.depth = self.parent.depth + 1
        else:
            self.cumulative_logprob = self.logprob
            self.depth = 0

    def add_child(self, token_id: int, logprob: float, token_text: Optional[str] = None) -> TreeNode:
        child = TreeNode(
            token_id=token_id,
            logprob=logprob,
            parent=self,
            token_text=token_text,
        )
        self.children.append(child)
        return child

    def get_path_from_root(self) -> List[TreeNode]:
        """Returns the list of nodes from the child of the root down to this node."""
        path = []
        curr: Optional[TreeNode] = self
        while curr is not None and curr.parent is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    def get_token_ids_from_root(self) -> List[int]:
        return [node.token_id for node in self.get_path_from_root()]

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class CautiousTree:
    """Manages the tree exploration, path perplexity evaluation, commitment, and pruning.

    Parameters:
        breadth: The branching factor B (candidate tokens sampled per frontier node).
        depth: The lookahead depth D of the tree.
        eos_token_id: The end-of-sequence token ID.
    """

    def __init__(self, breadth: int = 3, depth: int = 3, eos_token_id: Optional[int] = None):
        if breadth < 1:
            raise ValueError(f"breadth must be >= 1, got {breadth}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.breadth = breadth
        self.depth = depth
        self.eos_token_id = eos_token_id

        # Root node represents the anchor (committed prefix)
        self.root = TreeNode(token_id=-1, logprob=0.0)
        self.committed_tokens: List[int] = []

    def get_leaves(self) -> List[TreeNode]:
        """Finds all current leaf (frontier) nodes under self.root."""
        leaves: List[TreeNode] = []

        def _traverse(node: TreeNode):
            if node.is_leaf():
                # Root itself is not a valid leaf if it has children
                if node is not self.root:
                    leaves.append(node)
            else:
                for child in node.children:
                    _traverse(child)

        _traverse(self.root)
        return leaves if leaves else ([self.root] if self.root.is_leaf() else [])

    def current_max_depth(self) -> int:
        """Returns the maximum depth of any leaf relative to the current root."""
        leaves = self.get_leaves()
        if not leaves or leaves == [self.root]:
            return 0
        return max(leaf.depth for leaf in leaves)

    def get_all_root_to_leaf_paths(self) -> List[List[TreeNode]]:
        """Returns all complete paths from immediate children of root to leaves."""
        leaves = self.get_leaves()
        if not leaves or leaves == [self.root]:
            return []
        return [leaf.get_path_from_root() for leaf in leaves]

    @staticmethod
    def compute_path_perplexity(path: List[TreeNode]) -> float:
        """Computes the average perplexity of a path:
        PPL(p) = exp(-1/|p| * sum_{t=1}^{|p|} log p(y_t))
        """
        if not path:
            return float("inf")
        sum_logprob = sum(node.logprob for node in path)
        avg_neg_logprob = -sum_logprob / len(path)
        try:
            return math.exp(avg_neg_logprob)
        except OverflowError:
            return float("inf")

    def select_best_path(self) -> Tuple[List[TreeNode], float]:
        """Selects the path with the lowest average perplexity among all root-to-leaf paths."""
        paths = self.get_all_root_to_leaf_paths()
        if not paths:
            raise RuntimeError("No paths found in the tree to evaluate.")

        best_path = None
        min_ppl = float("inf")

        for path in paths:
            ppl = self.compute_path_perplexity(path)
            if ppl < min_ppl:
                min_ppl = ppl
                best_path = path

        assert best_path is not None
        return best_path, min_ppl

    def commit_and_prune(self) -> Tuple[int, Optional[str], float]:
        """Evaluates the paths, fixes the first token of the best path,

        prunes all competing branches, and re-roots the tree.

        Returns:
            committed_token_id: The token ID that was committed.
            committed_token_text: The text for the token if available.
            min_ppl: The perplexity of the winning path.
        """
        best_path, min_ppl = self.select_best_path()

        # The first token after the root in the winning path
        winning_first_node = best_path[0]
        committed_token_id = winning_first_node.token_id
        committed_text = winning_first_node.token_text

        # Commit token to global history
        self.committed_tokens.append(committed_token_id)

        # Prune all siblings of winning_first_node
        self.root.children = [winning_first_node]

        # Re-root at winning_first_node
        self.root = winning_first_node
        self.root.parent = None
        self._recalculate_depths(self.root, 0)

        return committed_token_id, committed_text, min_ppl

    def _recalculate_depths(self, node: TreeNode, current_depth: int):
        node.depth = current_depth
        for child in node.children:
            self._recalculate_depths(child, current_depth + 1)

    def total_nodes(self) -> int:
        """Returns total number of nodes in the active tree including root."""
        count = 0

        def _count(node: TreeNode):
            nonlocal count
            count += 1
            for child in node.children:
                _count(child)

        _count(self.root)
        return count
