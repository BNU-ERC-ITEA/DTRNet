"""
CharWiseVerifyLabelEncode (v4):
  Encodes labels for the CharWiseVerify pipeline.
  
  Key difference from v2: IDS labels are per-character, not whole-line.
  Each character gets its own IDS sub-sequence as a separate AR target.

  Output fields:
    - label:                [2 + max_text_len] text token ids (BOS + content + EOS + PAD)
    - length:               scalar, text content length
    - per_char_ids_labels:  [max_text_len, max_single_char_ids_len+2] per-char IDS (BOS + content + EOS + PAD)
    - per_char_ids_lengths: [max_text_len] per-char IDS content lengths
"""

import re

import numpy as np

from tools.utils.ids_to_tree import load_char_to_ids
from .ctc_label_encode import BaseRecLabelEncode


def _load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip("\n\r") for ln in f if ln.strip("\n\r")]


_CDP_RE = re.compile(r"&CDP-[0-9A-Fa-f]+;")


def _tokenize_ids(ids_str: str):
    s = str(ids_str).strip()
    if not s:
        return []
    if s.startswith("<") and s.endswith(">") and len(s) >= 3:
        return [s]
    toks = []
    i = 0
    while i < len(s):
        m = _CDP_RE.match(s, i)
        if m is not None:
            toks.append(m.group(0))
            i = m.end()
        else:
            toks.append(s[i])
            i += 1
    return toks


class CharWiseVerifyLabelEncode(BaseRecLabelEncode):

    def __init__(
        self,
        max_text_length,
        max_single_char_ids_len,
        character_dict_path,
        ids_dict_path,
        char_to_ids_path,
        use_space_char=False,
        **kwargs,
    ):
        super().__init__(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            lower=kwargs.get("lower", False),
        )

        # --- text vocab (from BaseRecLabelEncode)
        self.text_dict = self.dict
        self.text_max_len = max_text_length
        self.text_PAD = self.text_dict["<pad>"]
        self.text_SOS = self.text_dict["<sos>"]
        self.text_EOS = self.text_dict["<eos>"]
        self.text_UNK = self.text_dict["<unk>"]

        # --- ids vocab (independent)
        ids_chars = _load_list(ids_dict_path)
        if use_space_char and " " not in ids_chars:
            ids_chars.append(" ")

        ids_chars = ["<pad>", "<sos>", "<eos>", "<unk>"] + ids_chars
        self.ids_dict = {ch: i for i, ch in enumerate(ids_chars)}
        self.ids_PAD = self.ids_dict["<pad>"]
        self.ids_SOS = self.ids_dict["<sos>"]
        self.ids_EOS = self.ids_dict["<eos>"]
        self.ids_UNK = self.ids_dict["<unk>"]

        self.use_space_char = use_space_char
        self.max_single_char_ids_len = max_single_char_ids_len

        # char -> IDS string
        self.char2ids = load_char_to_ids(char_to_ids_path)

    def add_special_char(self, dict_character):
        return ["<pad>", "<sos>", "<eos>", "<unk>"] + dict_character

    def __call__(self, data):
        text = data["label"]
        if self.lower:
            text = text.lower()

        # ========== 1) Text branch ==========
        text_tokens = list(text)
        text_ids = [
            self.text_dict.get(ch, self.text_UNK) for ch in text_tokens
        ]
        text_len = min(len(text_ids), self.text_max_len)
        label_text = [self.text_SOS] + text_ids[:text_len] + [self.text_EOS]
        total_text = 2 + self.text_max_len
        if len(label_text) < total_text:
            label_text += [self.text_PAD] * (total_text - len(label_text))

        data["label"] = np.array(label_text, dtype=np.int64)
        data["length"] = np.array(text_len, dtype=np.int64)

        # ========== 2) Per-character IDS labels ==========
        # For each text character, encode its IDS as a separate AR sequence
        ids_label_len = self.max_single_char_ids_len + 2  # BOS + content + EOS
        per_char_ids_labels = np.zeros(
            (self.text_max_len, ids_label_len), dtype=np.int64,
        )  # PAD = 0

        per_char_ids_lengths = np.zeros(self.text_max_len, dtype=np.int64)

        for ci, ch in enumerate(text_tokens[:text_len]):
            # Get IDS string for this character
            ids_str = self.char2ids.get(ch)
            if ids_str is None:
                ids_str = ch if ch in self.ids_dict else "<unk>"

            # Tokenize IDS string:
            # - keep <...> as one token
            # - keep &CDP-xxxx; as one token
            ids_tokens = _tokenize_ids(ids_str)

            # Map to IDS vocab ids
            ids_token_ids = [
                self.ids_dict.get(tok, self.ids_UNK) for tok in ids_tokens
            ]

            # Truncate to max_single_char_ids_len
            ids_content_len = min(len(ids_token_ids), self.max_single_char_ids_len)

            # Build AR sequence: [BOS, ids_tok1, ..., ids_tokN, EOS, PAD...]
            seq = [self.ids_SOS] + ids_token_ids[:ids_content_len] + [self.ids_EOS]
            if len(seq) < ids_label_len:
                seq += [self.ids_PAD] * (ids_label_len - len(seq))

            per_char_ids_labels[ci, :] = np.array(seq[:ids_label_len], dtype=np.int64)
            per_char_ids_lengths[ci] = ids_content_len

        data["per_char_ids_labels"] = per_char_ids_labels
        data["per_char_ids_lengths"] = per_char_ids_lengths

        return data


# python -m openrec.preprocess.charwise_verify_label_encode
if __name__ == "__main__":
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)
    print("Testing CharWiseVerifyLabelEncode")
    print("=" * 60)

    max_text_len = 15
    max_single_char_ids_len = 15
    char_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    char_to_ids_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_to_ids.txt")

    for p in [char_dict_path, ids_dict_path, char_to_ids_path]:
        assert os.path.exists(p), f"File not found: {p}"

    try:
        encoder = CharWiseVerifyLabelEncode(
            max_text_length=max_text_len,
            max_single_char_ids_len=max_single_char_ids_len,
            character_dict_path=char_dict_path,
            ids_dict_path=ids_dict_path,
            char_to_ids_path=char_to_ids_path,
            use_space_char=True,
        )
        print(f"[OK] Encoder initialized. text_vocab={len(encoder.text_dict)}, ids_vocab={len(encoder.ids_dict)}")

        test_labels = ["我的", "你好世界", "一到春天，", "赢", "test", "a"]
        ids_label_len = max_single_char_ids_len + 2  # BOS + content + EOS

        for label in test_labels:
            data = {"label": label}
            result = encoder(data)

            text_label = result["label"]
            text_length = int(result["length"])
            per_char_ids = result["per_char_ids_labels"]
            per_char_ids_lens = result["per_char_ids_lengths"]

            print(f"\n--- label='{label}' ---")
            print(f"  text_label shape: {text_label.shape}, expected ({2 + max_text_len},)")
            print(f"  text_length: {text_length}")
            print(f"  per_char_ids shape: {per_char_ids.shape}, expected ({max_text_len}, {ids_label_len})")
            print(f"  per_char_ids_lens shape: {per_char_ids_lens.shape}, expected ({max_text_len},)")

            assert text_label.shape == (2 + max_text_len,), f"text_label shape mismatch"
            assert per_char_ids.shape == (max_text_len, ids_label_len), f"per_char_ids shape mismatch"
            assert per_char_ids_lens.shape == (max_text_len,), f"per_char_ids_lens shape mismatch"

            # Check BOS/EOS structure for each valid char
            for ci in range(text_length):
                ch = label[ci]
                ids_seq = per_char_ids[ci]
                ids_len = int(per_char_ids_lens[ci])
                assert ids_seq[0] == encoder.ids_SOS, f"char {ci} missing BOS"
                assert ids_seq[1 + ids_len] == encoder.ids_EOS, f"char {ci} missing EOS"
                # Decode back for display
                inv_ids = {v: k for k, v in encoder.ids_dict.items()}
                decoded = "".join([inv_ids.get(int(ids_seq[1+t]), "?") for t in range(ids_len)])
                print(f"  char[{ci}]='{ch}': ids_len={ids_len}, IDS='{decoded}'")

            # Padding chars should be all zeros
            for ci in range(text_length, max_text_len):
                assert np.all(per_char_ids[ci] == 0), f"Padding char {ci} not all zeros"
                assert per_char_ids_lens[ci] == 0, f"Padding char {ci} ids_len != 0"

        print("\n" + "=" * 60)
        print("[PASS] All CharWiseVerifyLabelEncode tests passed!")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {e}")
