"""Utilities to convert IDS prefix strings into tree supervision labels.

Core output: tokens, parents, relations
- tokens: list of IDS tokens (operators and radicals) after optional <sos>/<eos>.
- parents: same length as tokens; 0-based parent index; root has parent -1.
- relations: relation label from parent to node; root uses "PAD".

Intended usage: call ids_to_tree_supervision() inside your dataset preprocessing
or collate step to obtain structural supervision compatible with TAMER-style
parent-index loss. If you only need parent indices, ignore relations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .ids_syntax import DEFAULT_IDC_ARITY

# Relation mapping per IDC. You can extend or refine as needed.
REL_MAP: Dict[str, List[str]] = {
    "⿰": ["LEFT", "RIGHT"],
    "⿱": ["ABOVE", "BELOW"],
    "⿲": ["LEFT", "MIDDLE", "RIGHT"],
    "⿳": ["TOP", "MIDDLE", "BOTTOM"],
    "⿴": ["OUTER", "INNER"],
    "⿵": ["OUTER", "INNER"],
    "⿶": ["OUTER", "INNER"],
    "⿷": ["OUTER", "INNER"],
    "⿸": ["OUTER", "INNER"],
    "⿹": ["OUTER", "INNER"],
    "⿺": ["OUTER", "INNER"],
    "⿻": ["OUTER", "INNER"],
}


@dataclass
class TreeSupervision:
    tokens: List[str]
    parents: List[int]
    relations: List[str]


def _parse_prefix(
    seq: Sequence[str],
    idx: int,
    parent_idx: int,
    rel_from_parent: str,
    tokens: List[str],
    parents: List[int],
    relations: List[str],
    arity_map: Dict[str, int],
    rel_map: Dict[str, List[str]],
) -> int:
    """Recursive descent on a prefix IDS sequence.

    Returns the next unread position.
    """
    if idx >= len(seq):
        raise ValueError(f"Prefix parse out of range at idx={idx}, seq_len={len(seq)}")

    cur_idx = len(tokens)
    tok = seq[idx]
    tokens.append(tok)
    parents.append(parent_idx)
    relations.append(rel_from_parent if parent_idx != -1 else "PAD")

    arity = arity_map.get(tok, 0)
    if arity == 0:
        return idx + 1

    rel_list = rel_map.get(tok)
    if rel_list is None:
        # Fallback deterministic labels to keep length consistent.
        rel_list = [f"CH{k}" for k in range(arity)]
    elif len(rel_list) != arity:
        raise ValueError(f"Relation list size mismatch for token {tok}: {rel_list}")

    nxt = idx + 1
    for k in range(arity):
        nxt = _parse_prefix(
            seq,
            nxt,
            cur_idx,
            rel_list[k],
            tokens,
            parents,
            relations,
            arity_map,
            rel_map,
        )
    return nxt


def ids_to_tree_supervision(
    ids_seq: str,
    add_sos: bool = True,
    add_eos: bool = False,
    arity_map: Optional[Dict[str, int]] = None,
    rel_map: Optional[Dict[str, List[str]]] = None,
) -> TreeSupervision:
    """Convert a single IDS prefix string into TreeSupervision.

    Args:
        ids_seq: prefix IDS string, e.g., "⿱丿⿻亅⿹戈一".
        add_sos: prepend <sos> token as root.
        add_eos: append <eos> token (optional; seldom needed for tree).
        arity_map: override IDC arities; defaults to DEFAULT_IDC_ARITY.
        rel_map: override relation labels; defaults to REL_MAP.

    Raises:
        ValueError when the sequence is not fully consumed (invalid IDS).
    """
    if arity_map is None:
        arity_map = DEFAULT_IDC_ARITY
    if rel_map is None:
        rel_map = REL_MAP

    # Treat bracketed placeholders (e.g., <unk>) as single tokens to avoid
    # splitting into characters that cannot be parsed as IDS operators/radicals.
    if ids_seq.startswith("<") and ids_seq.endswith(">"):
        seq: List[str] = [ids_seq]
    else:
        seq: List[str] = list(ids_seq)
    if add_eos:
        seq.append("<eos>")

    # If sequence is a single IDC operator without operands, downgrade it to a leaf
    # to avoid runaway recursion. This can happen when raw text contains an IDC symbol
    # as a character but no structural expansion is available.
    local_arity = arity_map
    if len(seq) == 1 and arity_map.get(seq[0], 0) > 0:
        local_arity = dict(arity_map)
        local_arity[seq[0]] = 0

    tokens: List[str] = []
    parents: List[int] = []
    relations: List[str] = []

    if add_sos:
        # Create a virtual root first, then parse the remaining sequence as its single child subtree.
        root_idx = len(tokens)
        tokens.append("<sos>")
        parents.append(-1)
        relations.append("PAD")
        end_pos = _parse_prefix(
            seq=seq,
            idx=0,
            parent_idx=root_idx,
            rel_from_parent="CH0",
            tokens=tokens,
            parents=parents,
            relations=relations,
            arity_map=local_arity,
            rel_map=rel_map,
        )
    else:
        end_pos = _parse_prefix(
            seq=seq,
            idx=0,
            parent_idx=-1,
            rel_from_parent="PAD",
            tokens=tokens,
            parents=parents,
            relations=relations,
            arity_map=local_arity,
            rel_map=rel_map,
        )

    if end_pos != len(seq):
        raise ValueError(f"IDS not fully consumed: stop@{end_pos}/{len(seq)} for {ids_seq}")

    return TreeSupervision(tokens=tokens, parents=parents, relations=relations)


def load_char_to_ids(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ch, ids = line.split("\t", 1)
        mapping[ch] = ids
    return mapping


def char_sequence_to_forest(
    text: str,
    char2ids: Dict[str, str],
    add_sos: bool = True,
    add_eos: bool = False,
    arity_map: Optional[Dict[str, int]] = None,
    rel_map: Optional[Dict[str, List[str]]] = None,
) -> List[TreeSupervision]:
    """Convert a string of characters into per-character tree supervisions."""
    forest: List[TreeSupervision] = []
    for ch in text:
        ids = char2ids.get(ch)
        if ids is None:
            # Fallback to a leaf-only node when IDS is missing.
            sup = ids_to_tree_supervision(ch, add_sos=add_sos, add_eos=add_eos, arity_map=arity_map, rel_map=rel_map)
        else:
            sup = ids_to_tree_supervision(ids, add_sos=add_sos, add_eos=add_eos, arity_map=arity_map, rel_map=rel_map)
        forest.append(sup)
    return forest


def _demo():
    char2ids = load_char_to_ids("tools/utils/dict/visual_c3_ids/char_to_ids.txt")
    sample_char = "我"
    ids_seq = char2ids[sample_char]
    sup = ids_to_tree_supervision(ids_seq)
    print("tokens:   ", sup.tokens)
    print("parents:  ", sup.parents)
    print("relations:", sup.relations)


if __name__ == "__main__":
    _demo()
