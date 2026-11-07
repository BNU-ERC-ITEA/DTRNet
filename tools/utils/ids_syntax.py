"""
Lightweight IDS syntax validator for prefix-notation IDS strings.
- Uses a need-count (arity) rule to ensure IDC operands are satisfied.
- No dependencies on openrec; safe to import from any script.
"""
from typing import Iterable, List, Optional, Sequence, Tuple

# Default IDC arity map per Unicode IDS definition.
DEFAULT_IDC_ARITY = {
    "⿲": 3,
    "⿳": 3,
    "⿰": 2,
    "⿱": 2,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
    "⿼": 2,
    "⿽": 2,
    # Optional unary operators; leave in map but they can be pruned by vocab filtering.
    "⿾": 1,
    "⿿": 1,
}


def build_idc_arity_for_vocab(vocab: Optional[Iterable[str]]) -> dict:
    """
    Filter DEFAULT_IDC_ARITY by a given vocabulary (e.g., minimal_ids_dict.txt lines).
    If vocab is None, return the full map.
    """
    if vocab is None:
        return DEFAULT_IDC_ARITY
    vocab_set = set(vocab)
    return {k: v for k, v in DEFAULT_IDC_ARITY.items() if k in vocab_set}


def validate_ids_prefix(
    seq: Sequence[str],
    idc_arity: Optional[dict] = None,
    require_closed: bool = True,
) -> Tuple[bool, int, Optional[int]]:
    """
    Validate an IDS sequence (prefix notation) via need-count.

    Args:
        seq: iterable of tokens (string already split to characters).
        idc_arity: map from IDC char to arity; defaults to DEFAULT_IDC_ARITY.
        require_closed: when True, need must be 0 at end (complete tree).

    Returns:
        (is_valid, final_need, first_invalid_index)
        - first_invalid_index is set when need goes negative (over-close).
        - If require_closed and final_need > 0 (missing operands), first_invalid_index is len(seq)
          to indicate the truncation happens after the last token.
    """
    if idc_arity is None:
        idc_arity = DEFAULT_IDC_ARITY
    need = 1
    first_bad: Optional[int] = None
    for idx, ch in enumerate(seq):
        arity = idc_arity.get(ch, 0)
        need = need - 1 + arity
        if need < 0 and first_bad is None:
            first_bad = idx
            break
    if require_closed and need > 0 and first_bad is None:
        # Missing operands: mark the "bad" position as just after the last token
        first_bad = len(seq)
    is_valid = (need == 0) if require_closed else (need >= 0)
    return is_valid, need, first_bad


def pick_best_legal(
    candidates: List[Tuple[str, float]],
    idc_arity: Optional[dict] = None,
    require_closed: bool = True,
) -> Optional[Tuple[str, float]]:
    """
    Given a list of (sequence, score), return the first legal one.
    Assumes candidates are sorted by descending score.
    """
    if idc_arity is None:
        idc_arity = DEFAULT_IDC_ARITY
    for seq, score in candidates:
        ok, need, _ = validate_ids_prefix(seq, idc_arity, require_closed)
        if ok:
            return seq, score
    return None


def load_vocab_from_file(path: str) -> List[str]:
    """Load one token per line vocabulary file (e.g., minimal_ids_dict.txt)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # Example usage
    test_seq = "⿰亻⿱𠂇⿲口口口"
    test_seq = "⿰口⿵⿰丿戈一口"
    tokens = [tok for tok in str(test_seq).strip().split() if tok]
    seq_ok = True
    for tok in tokens:
        is_valid, need, first_bad = validate_ids_prefix([ch for ch in tok if not ch.isspace()], idc_arity=DEFAULT_IDC_ARITY, require_closed=True)
        if not is_valid:
            seq_ok = False
        print(f"tok: {tok}")
        print(f"Is valid: {is_valid}, Final need: {need}, First bad index: {first_bad}")
    if seq_ok:
        print(f"Sequence: {test_seq} is valid.")
    else:
        print(f"Sequence: {test_seq} is invalid.")