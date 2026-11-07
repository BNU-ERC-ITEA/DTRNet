"""
CharWiseVerifyPostProcess:
  Decodes outputs from the CharWiseVerify pipeline.

  Training: returns (text_preds, text_labels), (ids_preds, ids_labels) for metric
  Inference: returns text predictions + per-char IDS + error detection results
"""

import numpy as np
import torch
from .ctc_postprocess import BaseRecLabelDecode


class CharWiseVerifyPostProcess:

    def __init__(
        self,
        text_character_dict_path,
        ids_character_dict_path,
        char_to_ids_path=None,
        use_space_char=True,
        **kwargs,
    ):
        self.text_decoder = BaseRecLabelDecode(text_character_dict_path, use_space_char)
        self.ids_decoder = BaseRecLabelDecode(ids_character_dict_path, use_space_char)

        self.text_decoder.character = ["<pad>", "<sos>", "<eos>", "<unk>"] + self.text_decoder.character
        self.ids_decoder.character = ["<pad>", "<sos>", "<eos>", "<unk>"] + self.ids_decoder.character

        self.text_decoder.get_ignored_tokens = lambda: [0, 1, 2]
        self.ids_decoder.get_ignored_tokens = lambda: [0, 1, 2]
        self.character = self.text_decoder.character

        self.use_space_char = use_space_char

        self.ids2char = {}
        self.char2ids = {}
        if char_to_ids_path is not None:
            with open(char_to_ids_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        char, ids_seq = parts[0], parts[1].strip()
                        self.ids2char[ids_seq] = char
                        self.char2ids[char] = ids_seq

    def get_character_num(self):
        return len(self.text_decoder.character)

    def get_ignored_tokens(self):
        return self.text_decoder.get_ignored_tokens()

    def map_ids_to_text(self, ids_seq_str):
        """Map IDS string to text via ids2char lookup. Unmapped → 'X'."""
        if ids_seq_str is None:
            return ""
        ids_seq_str = str(ids_seq_str)
        if self.use_space_char:
            segs = ids_seq_str.split(" ")
        else:
            segs = [ids_seq_str]
        res = []
        for s in segs:
            if not s:
                continue
            res.append(self.ids2char.get(s, "X"))
        return "".join(res)

    def __call__(self, preds, batch=None, training=False, *args, **kwargs):
        if isinstance(preds, dict) and "res" in preds:
            preds = preds["res"]
        if training:
            return self._process_training(preds, batch)
        else:
            return self._process_inference(preds, batch)

    def _process_training(self, preds, batch):
        """Process training outputs for metric computation."""
        logits_text = preds[0]
        logits_ids = preds[1]

        if isinstance(logits_text, torch.Tensor):
            logits_text = logits_text.detach().cpu().numpy()

        text_dec = self._decode_text_ar(logits_text, self.text_decoder)

        label_text = []
        if batch is not None and len(batch) > 1:
            label_text = self._decode_label_text(batch[1], self.text_decoder)

        # CTC: logits_ids = (ctc_loss, all_char_ids_train)
        all_char_ids_train = logits_ids[1]
        ids_dec = self._reconstruct_ids_from_decoded_lists(all_char_ids_train)

        label_ids = self._reconstruct_label_ids(batch)

        return [(text_dec, label_text), (ids_dec, label_ids)]

    def _process_inference(self, preds, batch):
        """Process inference outputs."""
        probs_text, all_char_ids, text_len_pred = preds

        if isinstance(probs_text, torch.Tensor):
            probs_text = probs_text.detach().cpu().numpy()

        text_dec = self._decode_text_ar(probs_text, self.text_decoder)

        B = len(all_char_ids)
        ids_dec = []
        per_char_ids_strs = []
        error_flags = []

        for b in range(B):
            text_str = text_dec[b][0] if b < len(text_dec) else ""
            char_ids_list = all_char_ids[b]
            char_count = len(char_ids_list)

            char_ids_strs = []
            char_errors = []
            for ci in range(char_count):
                ids_tokens = char_ids_list[ci] if ci < len(char_ids_list) else []
                ids_str = self._ids_token_ids_to_str(ids_tokens)
                char_ids_strs.append(ids_str)

                is_error = False
                if ci < len(text_str):
                    pred_char = text_str[ci]
                    canonical_ids = self.char2ids.get(pred_char, None)
                    if canonical_ids is not None:
                        is_error = (ids_str != canonical_ids)
                    else:
                        is_error = True
                char_errors.append(is_error)

            full_ids_str = " ".join(char_ids_strs) if char_ids_strs else ""
            ids_dec.append((full_ids_str, 1.0))
            per_char_ids_strs.append(char_ids_strs)
            error_flags.append(char_errors)

        if batch is not None:
            label_text = []
            if len(batch) > 1:
                label_text = self._decode_label_text(batch[1], self.text_decoder)
            label_ids = self._reconstruct_label_ids(batch)
            return [(text_dec, label_text), (ids_dec, label_ids)]

        return {
            "text": text_dec,
            "ids": ids_dec,
            "per_char_ids": per_char_ids_strs,
            "error_flags": error_flags,
        }

    def _ids_token_ids_to_str(self, token_ids):
        """Convert IDS token id list to string."""
        chars = []
        for tid in token_ids:
            tid = int(tid)
            if tid in (0, 1, 2):
                continue
            if tid < len(self.ids_decoder.character):
                chars.append(self.ids_decoder.character[tid])
        return "".join(chars)

    def _reconstruct_ids_from_decoded_lists(self, all_char_ids):
        """Reconstruct per-sample IDS string from pre-decoded token ID lists."""
        results = []
        for b in range(len(all_char_ids)):
            char_ids_list = all_char_ids[b]
            char_ids_strs = []
            for ci in range(len(char_ids_list)):
                ids_tokens = char_ids_list[ci] if ci < len(char_ids_list) else []
                ids_str = self._ids_token_ids_to_str(ids_tokens)
                char_ids_strs.append(ids_str)
            full_ids = " ".join(char_ids_strs)
            results.append((full_ids, 1.0))
        return results

    def _reconstruct_label_ids(self, batch):
        """Reconstruct ground truth IDS strings from per-char labels."""
        if batch is None or len(batch) < 5:
            return []

        per_char_ids_labels = batch[3]
        per_char_ids_lengths = batch[4]
        text_lengths = batch[2]

        if isinstance(per_char_ids_labels, torch.Tensor):
            per_char_ids_labels = per_char_ids_labels.detach().cpu().numpy()
        if isinstance(per_char_ids_lengths, torch.Tensor):
            per_char_ids_lengths = per_char_ids_lengths.detach().cpu().numpy()
        if isinstance(text_lengths, torch.Tensor):
            text_lengths = text_lengths.detach().cpu().numpy()

        B = per_char_ids_labels.shape[0]
        results = []
        for b in range(B):
            tlen = int(text_lengths[b])
            char_ids_strs = []
            for ci in range(tlen):
                ids_len = int(per_char_ids_lengths[b, ci])
                ids_seq = per_char_ids_labels[b, ci, 1:1 + ids_len]
                tokens = []
                for tid in ids_seq:
                    tid = int(tid)
                    if tid in (0, 1, 2):
                        continue
                    if tid < len(self.ids_decoder.character):
                        tokens.append(self.ids_decoder.character[tid])
                char_ids_strs.append("".join(tokens))
            full_ids = " ".join(char_ids_strs)
            results.append((full_ids, 1.0))
        return results

    # -------- text AR decode --------
    def _decode_text_ar(self, probs, decoder):
        preds_idx = probs.argmax(axis=-1)
        preds_prob = probs.max(axis=-1)
        preds_idx = np.where(preds_idx == 3, 0, preds_idx)
        return self._greedy_decode_with_eos(preds_idx, preds_prob, decoder)

    def _greedy_decode_with_eos(self, text_index, text_prob, decoder):
        result_list = []
        eos_id = 2
        for b in range(len(text_index)):
            chars, confs = [], []
            for t, idx in enumerate(text_index[b]):
                idx = int(idx)
                if idx in (0, 1):
                    continue
                if idx == eos_id:
                    break
                if idx < len(decoder.character):
                    chars.append(decoder.character[idx])
                    confs.append(text_prob[b][t])
            s = "".join(chars)
            conf = float(np.mean(confs)) if confs else 0.0
            result_list.append((s, conf))
        return result_list

    def _decode_label_text(self, labels, decoder):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = labels[:, 1:]
        return decoder.decode(labels)


# python -m openrec.postprocess.charwise_verify_postprocess
if __name__ == "__main__":
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)
    print("Testing CharWiseVerifyPostProcess")
    print("=" * 60)

    max_text_len = 15
    max_single_char_ids_len = 15
    text_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_dict_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    char_to_ids_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_to_ids.txt")

    for p in [text_dict_path, ids_dict_path, char_to_ids_path]:
        assert os.path.exists(p), f"Not found: {p}"

    from openrec.preprocess.charwise_verify_label_encode import CharWiseVerifyLabelEncode

    try:
        encoder = CharWiseVerifyLabelEncode(
            max_text_length=max_text_len,
            max_single_char_ids_len=max_single_char_ids_len,
            character_dict_path=text_dict_path,
            ids_dict_path=ids_dict_path,
            char_to_ids_path=char_to_ids_path,
            use_space_char=True,
        )

        postprocess = CharWiseVerifyPostProcess(
            text_character_dict_path=text_dict_path,
            ids_character_dict_path=ids_dict_path,
            char_to_ids_path=char_to_ids_path,
            use_space_char=True,
        )
        print(f"[OK] PostProcess initialized. text_vocab={len(postprocess.text_decoder.character)}, ids_vocab={len(postprocess.ids_decoder.character)}")
        print(f"     ids2char entries: {len(postprocess.ids2char)}, char2ids entries: {len(postprocess.char2ids)}")

        text_vocab_size = len(postprocess.text_decoder.character)
        ids_vocab_size = len(postprocess.ids_decoder.character)

        # ---- Test training mode (CTC format) ----
        print("\n--- Testing training mode ---")
        test_samples = ["我的", "你好世"]
        bs = len(test_samples)
        ids_label_len = max_single_char_ids_len + 2

        text_labels_list = []
        per_char_ids_list = []
        per_char_ids_lens_list = []
        text_lens_list = []

        for text in test_samples:
            data = {"label": text}
            encoder(data)
            text_labels_list.append(data["label"])
            text_lens_list.append(data["length"])
            per_char_ids_list.append(data["per_char_ids_labels"])
            per_char_ids_lens_list.append(data["per_char_ids_lengths"])

        text_labels_t = torch.from_numpy(np.stack(text_labels_list))
        text_lengths_t = torch.from_numpy(np.stack(text_lens_list))
        per_char_ids_t = torch.from_numpy(np.stack(per_char_ids_list))
        per_char_ids_lens_t = torch.from_numpy(np.stack(per_char_ids_lens_list))

        max_text = int(text_lengths_t.max().item())

        # Simulate perfect text logits
        logits_text = torch.full((bs, 2 + max_text, text_vocab_size), -100.0)
        for b in range(bs):
            for t in range(2 + int(text_lengths_t[b])):
                tgt = int(text_labels_t[b, t])
                if t < logits_text.shape[1]:
                    logits_text[b, t, tgt] = 100.0

        # Simulate CTC decoded IDS (perfect: extract GT token ids)
        all_char_ids_train = []
        for b in range(bs):
            tlen = int(text_lengths_t[b])
            char_ids = []
            for ci in range(tlen):
                ids_len = int(per_char_ids_lens_t[b, ci])
                ids_tokens = per_char_ids_t[b, ci, 1:1+ids_len].tolist()
                char_ids.append(ids_tokens)
            all_char_ids_train.append(char_ids)

        ctc_loss_dummy = torch.tensor(0.0)
        logits_ids = (ctc_loss_dummy, all_char_ids_train)

        char_feat = torch.zeros(bs, max_text, 256)
        grammar_penalty = torch.tensor(0.0)

        pred = (logits_text, logits_ids, char_feat, grammar_penalty, max_text)
        image_ph = torch.zeros(bs, 3, 32, 256)
        batch = [image_ph, text_labels_t, text_lengths_t, per_char_ids_t, per_char_ids_lens_t]

        result = postprocess(pred, batch, training=True)
        assert len(result) == 2

        text_preds, text_labels_dec = result[0]
        ids_preds, ids_labels_dec = result[1]

        print(f"  text_preds: {text_preds}")
        print(f"  text_labels: {text_labels_dec}")
        print(f"  ids_preds: {ids_preds}")
        print(f"  ids_labels: {ids_labels_dec}")

        for i, text in enumerate(test_samples):
            pred_text = text_preds[i][0]
            print(f"  Sample '{text}' -> pred='{pred_text}', match={pred_text == text}")
        print("  [OK] Training mode verified.")

        # ---- Test inference mode ----
        print("\n--- Testing inference mode (with batch for eval) ---")
        probs_text = torch.softmax(logits_text, dim=-1)
        text_len_pred = text_lengths_t.clone()
        pred_inf = (probs_text, all_char_ids_train, text_len_pred)

        result_inf = postprocess(pred_inf, batch, training=False)
        assert len(result_inf) == 2
        print(f"  text_preds: {result_inf[0][0]}")
        print(f"  ids_preds: {result_inf[1][0]}")
        print("  [OK] Inference/eval mode verified.")

        # ---- Test pure inference (no batch) ----
        print("\n--- Testing pure inference (error detection) ---")
        result_pure = postprocess(pred_inf, batch=None, training=False)
        assert isinstance(result_pure, dict)
        assert "text" in result_pure
        assert "error_flags" in result_pure
        print(f"  text: {result_pure['text']}")
        print(f"  per_char_ids: {result_pure['per_char_ids']}")
        print(f"  error_flags: {result_pure['error_flags']}")
        print("  [OK] Pure inference verified.")

        print("\n" + "=" * 60)
        print("[PASS] All CharWiseVerifyPostProcess tests passed!")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {e}")
