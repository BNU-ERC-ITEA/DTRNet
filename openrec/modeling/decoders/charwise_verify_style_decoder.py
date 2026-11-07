"""
CharWiseVerifyDecoder — Independent Locator + Query-Valid Head + Per-char CTC

Dual-branch decoder for handwritten Chinese text-line recognition with
per-character IDS (Ideographic Description Sequence) verification.

Text branch:
  Autoregressive TransformerDecoder — standard next-token prediction for
  text-line recognition (BOS → char₁ → ... → charₙ → EOS).

IDS verification branch:
    1. Independent char_locator (TransformerDecoder with ordinal position
         queries) extracts per-position visual features from encoder memory.
     Zero information from text branch → fully independent verification.
  2. Per-char CTC head: each char feature → K frames → TransformerEncoder
     → independent per-frame IDS token prediction via CTC.

Why CTC (not AR) for IDS:
  AR decoders self-correct via sequential generation: even on noisy/fake
  characters, AR "rounds" output to nearest legal IDS → always matches
  ids2char → never flags errors (Char_R < 3%).
  CTC predicts each frame independently — no self-correction. On fake
  characters, CTC produces "approximately correct" IDS that fails exact
  ids2char lookup → marked as error → high Char_R.

Grammar legality guarantees:
  Training: differentiable penalty on structurally illegal token probability
            at CTC emission frames (_ctc_grammar_penalty).
  Inference: grammar-constrained CTC greedy decode with need counter
             (_ctc_constrained_decode).
"""

from typing import Optional, Tuple, List, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tamer_decoder import (
    WordPosEnc,
    TransformerDecoderLayer,
    TransformerDecoder,
    AttentionRefinementModule,
)


class CharWiseVerifyStyleDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels=None,
        text_vocab_path: Optional[str] = None,
        ids_vocab_path: Optional[str] = None,
        char_to_ids_path: Optional[str] = None,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 64,
        cross_coverage: bool = False,
        self_coverage: bool = False,
        max_text_length: int = 25,
        use_space_char: bool = False,
        # Grammar constraints
        constrained_ctc_decode: bool = True,
        ids_syntax_max_need: int = 64,
        **kwargs,
    ):
        super().__init__()

        def _infer_vocab_size(path: Optional[str]) -> int:
            if path is None:
                raise ValueError("vocab_path is required to infer vocab size.")
            with open(path, "rb") as fin:
                count = len(fin.readlines())
            if use_space_char:
                count += 1
            return count + 4  # <pad><sos><eos><unk>

        text_vocab_size = out_channels if out_channels is not None else _infer_vocab_size(text_vocab_path)
        ids_vocab_size = _infer_vocab_size(ids_vocab_path)

        # Special tokens
        self.ignore_index = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.max_text_len = int(max_text_length)
        self.use_space_char = bool(use_space_char)
        self.text_vocab_size = text_vocab_size
        self.ids_vocab_size = ids_vocab_size

        # Grammar settings
        self.constrained_ctc_decode = bool(constrained_ctc_decode)
        self.ids_syntax_max_need = int(ids_syntax_max_need)

        def _to_bool(v, default=False):
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "y", "on")
            return bool(v)

        # Inference-only IDS-guided text calibration
        self.use_ids_bonus = _to_bool(kwargs.get("use_ids_bonus", False), default=False)
        self.ids_exact_bonus = float(kwargs.get("ids_exact_bonus", 1.0))
        self.ids_soft_bonus = float(kwargs.get("ids_soft_bonus", 0.25))

        # Master switch for IDS branch ablation.
        self.enable_ids_branch = _to_bool(kwargs.get("enable_ids_branch", True), default=True)

        # Query-valid head switch (prefix 1/0 supervision).
        self.use_valid_head = self.enable_ids_branch and _to_bool(kwargs.get("use_valid_head", True), default=True)
        self.use_ids_bonus = self.enable_ids_branch and self.use_ids_bonus

        # No threshold gates for IDS bonus: legality decides whether IDS can speak.
        # Confidence and NED similarity are used only as continuous weights.

        d_model = in_channels
        self.d_model = d_model

        # ===== Text branch (AR) =====
        self.text_embed = nn.Sequential(
            nn.Embedding(text_vocab_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.text_pos_enc = WordPosEnc(d_model)
        self.text_norm = nn.LayerNorm(d_model)

        arm_factory = (
            lambda: AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
            if (cross_coverage or self_coverage)
            else None
        )
        self.text_decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_decoder_layers,
            arm=arm_factory(),
        )
        self.proj_text = nn.Linear(d_model, text_vocab_size)

        # ===== IDS branch modules =====
        if self.enable_ids_branch:
            num_char_locator_layers = int(kwargs.get("num_char_locator_layers", 2))
            self.char_query_embed = nn.Embedding(max_text_length, d_model)
            self.char_query_pos_bias = nn.Parameter(
                torch.randn(1, max_text_length, d_model) * 0.02
            )
            nn.init.normal_(self.char_query_embed.weight, mean=0.0, std=0.02)
            char_loc_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.char_locator = nn.TransformerDecoder(
                char_loc_layer,
                num_layers=num_char_locator_layers,
            )
            self.char_visual_ln = nn.LayerNorm(d_model)

            if self.use_valid_head:
                self.char_valid_head = nn.Linear(d_model, 1)
            else:
                self.char_valid_head = None

            ctc_frames = int(kwargs.get("ctc_frames_per_char", 32))
            num_ctc_encoder_layers = int(kwargs.get("num_ctc_encoder_layers", 1))
            self.ctc_frames = ctc_frames
            self.ctc_blank_id = 0

            self.char_frame_expand = nn.Linear(d_model, ctc_frames * d_model)
            self.char_frame_pos = nn.Parameter(
                torch.randn(ctc_frames, d_model) * 0.02
            )

            ctc_enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.ctc_frame_encoder = nn.TransformerEncoder(
                ctc_enc_layer,
                num_layers=num_ctc_encoder_layers,
            )
            self.ctc_proj = nn.Linear(d_model, ids_vocab_size)

            self.ctc_loss_fn = nn.CTCLoss(
                blank=self.ctc_blank_id,
                reduction="mean",
                zero_infinity=True,
            )

        else:
            self.char_query_embed = None
            self.char_query_pos_bias = None
            self.char_locator = None
            self.char_visual_ln = None
            self.char_valid_head = None
            self.ctc_frames = 0
            self.ctc_blank_id = 0
            self.char_frame_expand = None
            self.char_frame_pos = None
            self.ctc_frame_encoder = None
            self.ctc_proj = None
            self.ctc_loss_fn = None

        # ===== Text vocab (for inference-time IDS bonus) =====
        self.text_tokens: Optional[List[str]] = None
        self.text_token2id: Optional[Dict[str, int]] = None
        if text_vocab_path is not None:
            text_chars = []
            with open(text_vocab_path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip("\n\r")
                    if s:
                        text_chars.append(s)
            if self.use_space_char and " " not in text_chars:
                text_chars.append(" ")
            self.text_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"] + text_chars
            self.text_token2id = {t: i for i, t in enumerate(self.text_tokens)}

        # ===== IDS vocab =====
        self.ids_tokens: Optional[List[str]] = None
        self.ids_token2id: Optional[Dict[str, int]] = None
        self.space_id: Optional[int] = None

        if ids_vocab_path is not None:
            ids_chars = []
            with open(ids_vocab_path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip("\n\r")
                    if s:
                        ids_chars.append(s)
            if self.use_space_char and " " not in ids_chars:
                ids_chars.append(" ")
            self.ids_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"] + ids_chars
            self.ids_token2id = {t: i for i, t in enumerate(self.ids_tokens)}
            self.space_id = self.ids_token2id.get(" ", None)

        # ===== IDC operator arity sets =====
        self._ids_unary_ops: Set[int] = set()
        self._ids_binary_ops: Set[int] = set()
        self._ids_trinary_ops: Set[int] = set()
        self._ids_leaf_ids: Set[int] = set()
        self._ids_operator_ids: Set[int] = set()
        self._init_ids_operator_sets()

        # ===== char_to_ids mapping =====
        self.char_to_ids_map: Optional[Dict[str, str]] = None
        if char_to_ids_path is not None:
            self.char_to_ids_map = {}
            with open(char_to_ids_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        self.char_to_ids_map[parts[0]] = parts[1].strip()

        self.ids_to_char_map: Optional[Dict[str, str]] = None
        self._char_ids_items: List[Tuple[str, str]] = []
        if self.char_to_ids_map is not None:
            self.ids_to_char_map = {}
            for ch, ids_str in self.char_to_ids_map.items():
                if ids_str:
                    self._char_ids_items.append((ch, ids_str))
                    if ids_str not in self.ids_to_char_map:
                        self.ids_to_char_map[ids_str] = ch

    # -------------------- Helpers --------------------
    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), True, dtype=torch.bool, device=device)
        m.triu_(1)
        return m

    def _prep_memory(
        self, x
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            memory:   [B, T, C]
            h:        feature height
            mem_mask: [B, T] with True = pad
            feat2d:   [B, H, W, C] or None
        """
        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                feat2d, mask2d = x  # feat2d: [B,H,W,C], mask2d: [B,H,W]
                b, h, w, c = feat2d.shape
                mem = feat2d.view(b, h * w, c)
                mem_mask = mask2d.reshape(b, h * w).bool()
                return mem, h, mem_mask, feat2d
            elif len(x) == 3:
                mem, hw, mem_mask = x
                h, w = hw
                feat2d = None
                if mem.dim() == 3 and mem.size(1) == h * w:
                    feat2d = mem.view(mem.size(0), h, w, mem.size(-1))
                return mem, h, mem_mask.bool() if mem_mask is not None else None, feat2d

        # fallback: already flattened memory [B,T,C]
        return x, 2, None, None

    @torch.no_grad()
    def _prefix_lengths_from_valid_logits(self, valid_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert per-query valid logits to prefix mask and length.

        Rule: valid logits > 0 means query is valid (1); then enforce
        monotonic prefix format (1...10...0) via cumulative product.

        Args:
            valid_logits: [B, max_chars]
        Returns:
            valid_len_pred: [B]
            valid_prefix_mask: [B, max_chars]
        """
        valid_mask = valid_logits > 0
        valid_prefix_mask = valid_mask.long().cumprod(dim=1).bool()
        valid_len_pred = valid_prefix_mask.long().sum(dim=1)
        return valid_len_pred, valid_prefix_mask

    def _decode_text(
        self,
        tgt: torch.Tensor,                    # [B,L]
        memory: torch.Tensor,                 # [B,Tenc,C]
        h: int,
        mem_mask: Optional[torch.Tensor],     # [B,Tenc]
    ) -> torch.Tensor:
        """
        Decode text autoregressively from encoder memory.
        Returns hidden states with shape [B,L,C].
        """
        device = memory.device
        pad_mask = (tgt == self.ignore_index)
        L = tgt.size(1)
        tgt_mask = self._causal_mask(L, device=device)

        out = self.text_embed(tgt)
        out = self.text_pos_enc(out)
        out = self.text_norm(out)
        out = out.transpose(0, 1)  # [L,B,C]

        mem_t = memory.transpose(0, 1)
        prev_attn = None
        for layer in self.text_decoder.layers:
            out, attn = layer(
                out,
                mem_t,
                arm=self.text_decoder.arm,
                prev_cross_attn=prev_attn,
                height=h,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask,
            )
            if self.text_decoder.arm is not None:
                prev_attn = attn

        out = self.text_decoder.norm(out)
        return out.transpose(0, 1)

    # -------------------- IDC operator sets --------------------
    def _init_ids_operator_sets(self):
        if self.ids_token2id is None:
            return
        unary = ["⿾", "⿿"]
        trinary = ["⿲", "⿳"]
        binary = ["⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻", "⿼", "⿽", "㇯"]

        specials = {self.ignore_index, self.bos_id, self.eos_id, self.unk_id}
        if self.space_id is not None:
            specials.add(self.space_id)

        for s in unary:
            if s in self.ids_token2id:
                self._ids_unary_ops.add(self.ids_token2id[s])
        for s in trinary:
            if s in self.ids_token2id:
                self._ids_trinary_ops.add(self.ids_token2id[s])
        for s in binary:
            if s in self.ids_token2id:
                self._ids_binary_ops.add(self.ids_token2id[s])

        self._ids_operator_ids = self._ids_unary_ops | self._ids_binary_ops | self._ids_trinary_ops
        for tok_id in range(self.ids_vocab_size):
            if tok_id not in specials and tok_id not in self._ids_operator_ids:
                self._ids_leaf_ids.add(tok_id)

    # -------------------- Independent char feature extraction --------------------
    def _extract_char_features(
        self,
        memory: torch.Tensor,
        mem_mask: Optional[torch.Tensor],
        num_chars: int,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Independent positional decoder for per-char visual feature extraction.

        Uses ordinal position queries that attend to encoder memory via
        cross-attention. Self-attention between queries enables coordinate
        ordering. No information from text branch.

        Padding query slots (beyond each sample's actual length) are masked
        via tgt_key_padding_mask to prevent them from participating in
        self-attention.

        Returns:
            char_feat: [B, num_chars, 1, d_model]
            valid_logits: [B, num_chars]
        """
        B = memory.size(0)
        pos_ids = torch.arange(num_chars, device=memory.device)
        queries = self.char_query_embed(pos_ids).unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.char_query_pos_bias[:, :num_chars, :]

        tgt_kpm = None
        if lengths is not None:
            idx = torch.arange(num_chars, device=memory.device).unsqueeze(0)
            tgt_kpm = idx >= lengths.unsqueeze(1)

        char_feat = self.char_locator(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_mask,
        )
        char_feat = self.char_visual_ln(char_feat)
        if self.char_valid_head is not None:
            valid_logits = self.char_valid_head(char_feat).squeeze(-1)
        else:
            valid_logits = char_feat.new_zeros((B, num_chars))
        return char_feat.unsqueeze(2), valid_logits

    # -------------------- Per-char CTC --------------------
    def _ctc_forward(
        self,
        char_feat: torch.Tensor,                  # [N, d_model]
    ) -> torch.Tensor:
        """
        Expand char feature to K frames, encode, and project to IDS token logits.
        """
        N, d = char_feat.shape
        K = self.ctc_frames

        frames = self.char_frame_expand(char_feat).view(N, K, d)
        frames = frames + self.char_frame_pos.unsqueeze(0)

        frames = self.ctc_frame_encoder(frames)
        ctc_logits = self.ctc_proj(frames)
        return ctc_logits

    def _ctc_loss(
        self,
        ctc_logits: torch.Tensor,    # [N, K, V_ids]
        ids_targets: torch.Tensor,   # [N, L_ids]
        ids_lengths: torch.Tensor,   # [N]
    ) -> torch.Tensor:
        """CTC loss for N valid characters."""
        N, K, V = ctc_logits.shape
        device = ctc_logits.device

        if N == 0:
            return torch.tensor(0.0, device=device)

        log_probs = F.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
        max_tgt_len = int(ids_lengths.max().item())
        targets = ids_targets[:, 1:1 + max_tgt_len].contiguous()
        input_lengths = torch.full((N,), K, dtype=torch.long, device=device)
        return self.ctc_loss_fn(log_probs, targets, input_lengths, ids_lengths.long())

    @torch.no_grad()
    def _ctc_greedy_decode(self, ctc_logits: torch.Tensor) -> List[List[int]]:
        """CTC greedy decode: argmax → collapse duplicates → remove blank."""
        pred_ids = ctc_logits.argmax(dim=-1)
        results = []
        for i in range(pred_ids.size(0)):
            tokens = pred_ids[i].tolist()
            collapsed = []
            prev = None
            for t in tokens:
                if t != prev:
                    collapsed.append(t)
                prev = t
            results.append([t for t in collapsed if t not in (0, 1, 2)])
        return results

    def _is_ids_token_legal(self, token_id: int, need: int) -> bool:
        """Check if a token is legal given current IDS grammar state."""
        if token_id in (0, 1, 2, 3):
            return False
        if need <= 0:
            return False
        if token_id in self._ids_binary_ops:
            return (need + 1) <= self.ids_syntax_max_need
        if token_id in self._ids_trinary_ops:
            return (need + 2) <= self.ids_syntax_max_need
        return True

    @torch.no_grad()
    def _ctc_constrained_decode(self, ctc_logits: torch.Tensor) -> List[List[int]]:
        """
        Grammar-constrained CTC greedy decode.

        At each emission point, check IDS grammar state (need counter).
        If argmax token is illegal, pick highest-scoring legal token.
        Ensures decoded IDS always forms a valid tree structure.
        """
        N, K, V = ctc_logits.shape
        results = []

        for i in range(N):
            need = 1
            prev = -1
            decoded = []

            for k in range(K):
                scores = ctc_logits[i, k]
                top = scores.argmax().item()

                if top == 0 or top == prev:
                    prev = top
                    continue

                if need <= 0:
                    prev = 0
                    continue

                if not self._is_ids_token_legal(top, need):
                    sorted_indices = scores.argsort(descending=True)
                    found = False
                    for idx_t in sorted_indices:
                        idx = idx_t.item()
                        if idx == 0 or idx == prev:
                            continue
                        if self._is_ids_token_legal(idx, need):
                            top = idx
                            found = True
                            break
                    if not found:
                        prev = 0
                        continue

                decoded.append(top)
                prev = top

                if top in self._ids_binary_ops:
                    need += 1
                elif top in self._ids_trinary_ops:
                    need += 2
                elif top in self._ids_unary_ops:
                    pass
                elif top not in (0, 1, 2, 3):
                    need -= 1

            results.append(decoded)
        return results

    def _ctc_grammar_penalty(self, ctc_logits: torch.Tensor) -> torch.Tensor:
        """
        Differentiable grammar penalty on CTC frame logits (training time).

        At each emission frame, penalize softmax probability on structurally
        illegal tokens. Fully vectorized — no Python loops over N or K.

        Design:
          1. Argmax determines emission frames and grammar state (detached,
             non-differentiable — treated as constant).
          2. Softmax probabilities on illegal tokens form the differentiable
             penalty (pushes probability mass toward legal tokens).
        """
        N, K, V = ctc_logits.shape
        device = ctc_logits.device

        if N == 0:
            return torch.tensor(0.0, device=device)

        # Identify emission frames
        pred_ids = ctc_logits.detach().argmax(dim=-1)
        is_blank = (pred_ids == 0)
        is_dup = torch.zeros_like(pred_ids, dtype=torch.bool)
        is_dup[:, 1:] = (pred_ids[:, 1:] == pred_ids[:, :-1])
        is_emission = ~is_blank & ~is_dup

        # Per-frame grammar delta
        delta = torch.zeros(N, K, device=device)
        for op_id in self._ids_binary_ops:
            delta += (pred_ids == op_id).float()
        for op_id in self._ids_trinary_ops:
            delta += (pred_ids == op_id).float() * 2
        is_special = (pred_ids == 0) | (pred_ids == 1) | (pred_ids == 2) | (pred_ids == 3)
        is_op = torch.zeros(N, K, dtype=torch.bool, device=device)
        for op_id in self._ids_operator_ids:
            is_op = is_op | (pred_ids == op_id)
        is_leaf_emission = is_emission & ~is_special & ~is_op
        delta = delta - is_leaf_emission.float()
        delta = delta * is_emission.float()

        # Cumulative need
        cum_delta = delta.cumsum(dim=1)
        need = torch.ones(N, K, device=device)
        need[:, 1:] = need[:, 1:] + cum_delta[:, :-1]

        # Penalty per emission frame
        probs = F.softmax(ctc_logits, dim=-1)
        emission_need_pos = is_emission & (need > 0)
        emission_need_done = is_emission & (need <= 0)

        penalty = torch.zeros(N, K, device=device)
        penalty += (1.0 - probs[:, :, 0]) * emission_need_done.float()
        penalty += probs[:, :, 1] * emission_need_pos.float()  # BOS
        penalty += probs[:, :, 2] * emission_need_pos.float()  # EOS
        penalty += probs[:, :, 3] * emission_need_pos.float()  # UNK

        binary_overflow = emission_need_pos & (need + 1 > self.ids_syntax_max_need)
        for op_id in self._ids_binary_ops:
            penalty += probs[:, :, op_id] * binary_overflow.float()

        trinary_overflow = emission_need_pos & (need + 2 > self.ids_syntax_max_need)
        for op_id in self._ids_trinary_ops:
            penalty += probs[:, :, op_id] * trinary_overflow.float()

        n_emissions = is_emission.float().sum().clamp_min(1.0)
        return penalty.sum() / n_emissions


    @torch.no_grad()
    def _ctc_char_confidence(self, ctc_logits: torch.Tensor) -> torch.Tensor:
        """
        Confidence of decoded IDS sequence.
        Average frame max-probability on emission frames only
        (blank / repeated collapsed frames are ignored).
        """
        if ctc_logits.numel() == 0:
            return ctc_logits.new_zeros((0,))

        probs = F.softmax(ctc_logits, dim=-1)
        pred = probs.argmax(dim=-1)  # [N,K]
        maxp = probs.max(dim=-1).values

        is_blank = pred.eq(self.ctc_blank_id)
        is_dup = torch.zeros_like(pred, dtype=torch.bool)
        is_dup[:, 1:] = pred[:, 1:].eq(pred[:, :-1])
        is_emission = ~is_blank & ~is_dup

        conf = ctc_logits.new_zeros((ctc_logits.size(0),))
        for i in range(ctc_logits.size(0)):
            mask = is_emission[i]
            if mask.any():
                conf[i] = maxp[i][mask].mean()
        return conf

    def _ids_list_to_string(self, ids_list: List[int]) -> str:
        if self.ids_tokens is None:
            return ""
        out = []
        for idx in ids_list:
            if 0 <= idx < len(self.ids_tokens):
                tok = self.ids_tokens[idx]
                if tok not in ("<pad>", "<sos>", "<eos>", "<unk>"):
                    out.append(tok)
        return "".join(out)

    def _is_complete_legal_ids(self, ids_list: List[int]) -> bool:
        if not ids_list:
            return False
        need = 1
        for j, token_id in enumerate(ids_list):
            if not self._is_ids_token_legal(token_id, need):
                return False
            if token_id in self._ids_binary_ops:
                need += 1
            elif token_id in self._ids_trinary_ops:
                need += 2
            elif token_id in self._ids_unary_ops:
                pass
            else:
                need -= 1
            if need <= 0 and j != len(ids_list) - 1:
                return False
        return need == 0

    def _normalized_edit_similarity(self, s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            c1 = s1[i - 1]
            for j in range(1, n + 1):
                tmp = dp[j]
                cost = 0 if c1 == s2[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        dist = dp[n]
        return max(0.0, 1.0 - dist / max(m, n))

    @torch.no_grad()
    def _build_ids_bonus(self, all_char_ids: List[List[List[int]]], all_char_conf: List[List[float]]) -> Optional[torch.Tensor]:
        if (not self.use_ids_bonus) or self.text_token2id is None or self.ids_to_char_map is None:
            return None

        B = len(all_char_ids)
        device = self.proj_text.weight.device
        bonus = torch.zeros(B, self.max_text_len, self.text_vocab_size, device=device)

        for b in range(B):
            n = min(len(all_char_ids[b]), self.max_text_len)
            for i in range(n):
                ids_list = all_char_ids[b][i]
                conf = float(all_char_conf[b][i]) if i < len(all_char_conf[b]) else 0.0
                if not self._is_complete_legal_ids(ids_list):
                    continue

                ids_str = self._ids_list_to_string(ids_list)
                if not ids_str:
                    continue

                mapped_char = self.ids_to_char_map.get(ids_str)
                if mapped_char is not None:
                    tok_id = self.text_token2id.get(mapped_char)
                    if tok_id is not None:
                        bonus[b, i, tok_id] += self.ids_exact_bonus * conf
                    continue

                if self.ids_soft_bonus <= 0.0:
                    continue

                best_char = None
                best_sim = -1.0
                for ch, ref_ids in self._char_ids_items:
                    sim = self._normalized_edit_similarity(ids_str, ref_ids)
                    if sim > best_sim:
                        best_sim = sim
                        best_char = ch
                if best_char is None:
                    continue
                tok_id = self.text_token2id.get(best_char)
                if tok_id is not None:
                    bonus[b, i, tok_id] += self.ids_soft_bonus * conf * float(best_sim)

        return bonus

    # -------------------- Forward (train) --------------------
    def forward_train(self, x, data):
        memory, h, mem_mask, _feat2d = self._prep_memory(x)
        text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths = data[:4]

        if not self.enable_ids_branch:
            max_text = int(text_lengths.max().item())
            tgt_text = text_labels[:, :2 + max_text]
            text_hidden = self._decode_text(
                tgt=tgt_text,
                memory=memory,
                h=h,
                mem_mask=mem_mask,
            )
            logits_text = self.proj_text(text_hidden)
            zero = logits_text.new_zeros(())
            B = memory.size(0)
            empty_ids = [[] for _ in range(B)]
            empty_char_feat = logits_text.new_zeros((B, self.max_text_len, 1, self.d_model))
            return logits_text, (zero, empty_ids, zero), empty_char_feat, zero, max_text

        # 1. IDS branch first: extract all query-wise char features
        max_chars = self.max_text_len
        char_feat, valid_logits = self._extract_char_features(memory, mem_mask, max_chars, lengths=None)
        char_feat_2d = char_feat[:, :, 0, :]  # [B,max_chars,d]

        # 1.5 Query-valid supervision (prefix 1/0)
        if self.use_valid_head:
            valid_targets = (
                torch.arange(max_chars, device=memory.device).unsqueeze(0)
                < text_lengths.unsqueeze(1)
            ).to(valid_logits.dtype)
            valid_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets)
        else:
            valid_loss = torch.tensor(0.0, device=memory.device)

        # 2. Per-char CTC on GT-valid positions only
        B = memory.size(0)
        device = memory.device

        ids_label_2d = per_char_ids_labels[:, :max_chars, :]
        ids_len_2d = per_char_ids_lengths[:, :max_chars]

        valid_char_mask = torch.arange(max_chars, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
        char_feat_valid = char_feat_2d[valid_char_mask]      # [N,d]
        ids_valid = ids_label_2d[valid_char_mask]            # [N,L]
        ids_len_valid = ids_len_2d[valid_char_mask]          # [N]

        ctc_logits_valid = self._ctc_forward(char_feat_valid)

        ids_ctc_loss = self._ctc_loss(ctc_logits_valid, ids_valid, ids_len_valid)

        grammar_penalty = self._ctc_grammar_penalty(ctc_logits_valid)

        with torch.no_grad():
            ids_decoded_valid = self._ctc_greedy_decode(ctc_logits_valid)

        all_char_ids_train: List[List[List[int]]] = []
        idx = 0
        for b in range(B):
            n = min(int(text_lengths[b].item()), max_chars)
            all_char_ids_train.append(ids_decoded_valid[idx:idx + n])
            idx += n

        # 3. Text AR

        max_text = int(text_lengths.max().item())
        tgt_text = text_labels[:, :2 + max_text]
        text_hidden = self._decode_text(
            tgt=tgt_text,
            memory=memory,
            h=h,
            mem_mask=mem_mask,
        )
        logits_text = self.proj_text(text_hidden)

        # Return independent losses:
        # - grammar_penalty as illegal_loss branch
        # - valid_loss as query-valid branch (inside logits_ids tuple)
        return logits_text, (ids_ctc_loss, all_char_ids_train, valid_loss), char_feat, grammar_penalty, max_text

    # -------------------- Forward (test) --------------------
    def forward_test(self, x):
        memory, h, mem_mask, _feat2d = self._prep_memory(x)
        B, _, _ = memory.shape
        device = memory.device

        if not self.enable_ids_branch:
            tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
            probs_text_steps = []

            for i in range(self.max_text_len + 1):
                hidden = self._decode_text(
                    tgt=tgt,
                    memory=memory,
                    h=h,
                    mem_mask=mem_mask,
                )

                logits_i = self.proj_text(hidden[:, -1:, :])
                probs_i = F.softmax(logits_i, dim=-1)
                probs_text_steps.append(probs_i)

                if i < self.max_text_len:
                    nxt = logits_i.squeeze(1).argmax(-1)
                    tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                    if (tgt == self.eos_id).any(dim=-1).all():
                        break

            probs_text = torch.cat(probs_text_steps, dim=1)

            pred_tokens = tgt[:, 1:]
            eos_mask = (pred_tokens == self.eos_id)
            has_eos = eos_mask.any(dim=1)
            first_eos = torch.zeros((B,), dtype=torch.long, device=device)
            if has_eos.any():
                first_eos = eos_mask.float().argmax(dim=1)
            text_len_pred = torch.where(
                has_eos,
                first_eos,
                torch.full((B,), pred_tokens.size(1), device=device, dtype=torch.long),
            )
            text_len_pred = torch.clamp(text_len_pred, min=0)

            all_char_ids = []
            for b in range(B):
                n = min(int(text_len_pred[b].item()), self.max_text_len)
                all_char_ids.append([[] for _ in range(n)])

            return probs_text, all_char_ids, text_len_pred

        # 1. Extract all query-wise char features once
        max_chars = self.max_text_len
        char_feat_all, valid_logits = self._extract_char_features(memory, mem_mask, max_chars, lengths=None)
        char_feat_2d = char_feat_all[:, :, 0, :]  # [B,max_chars,d]

        # 2. Predict valid character length from query-valid logits
        if self.use_valid_head:
            valid_len_pred, valid_char_mask = self._prefix_lengths_from_valid_logits(valid_logits)
        else:
            valid_len_pred = torch.full((B,), max_chars, dtype=torch.long, device=device)
            valid_char_mask = torch.ones((B, max_chars), dtype=torch.bool, device=device)
        valid_len_pred = torch.clamp(valid_len_pred, min=0, max=max_chars)

        char_feat_valid = char_feat_2d[valid_char_mask]  # [N,d]

        if char_feat_valid.numel() > 0:
            ctc_logits_valid = self._ctc_forward(char_feat_valid)
            char_conf_valid = self._ctc_char_confidence(ctc_logits_valid)

            if self.constrained_ctc_decode:
                all_ids_valid = self._ctc_constrained_decode(ctc_logits_valid)
            else:
                all_ids_valid = self._ctc_greedy_decode(ctc_logits_valid)
        else:
            char_conf_valid = torch.zeros((0,), device=device)
            all_ids_valid = []

        # Reconstruct per-sample IDS results using valid-head predicted valid length
        decoded_char_ids = []
        all_char_conf = []
        idx = 0
        for b in range(B):
            n = min(int(valid_len_pred[b].item()), max_chars)
            decoded_char_ids.append(all_ids_valid[idx:idx + n])
            all_char_conf.append(char_conf_valid[idx:idx + n].tolist())
            idx += n

        ids_bonus = self._build_ids_bonus(decoded_char_ids, all_char_conf)

        # 3. Text greedy AR with optional IDS bonus
        tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        probs_text_steps = []

        for i in range(self.max_text_len + 1):
            hidden = self._decode_text(
                tgt=tgt,
                memory=memory,
                h=h,
                mem_mask=mem_mask,
            )

            logits_i = self.proj_text(hidden[:, -1:, :])

            if ids_bonus is not None and i < self.max_text_len:
                base_top1 = logits_i.squeeze(1).argmax(dim=-1)
                step_bonus = ids_bonus[:, i, :].clone()
                step_bonus[base_top1.eq(self.eos_id)] = 0.0
                logits_i = logits_i + step_bonus.unsqueeze(1)

            probs_i = F.softmax(logits_i, dim=-1)
            probs_text_steps.append(probs_i)

            if i < self.max_text_len:
                nxt = logits_i.squeeze(1).argmax(-1)
                tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                if (tgt == self.eos_id).any(dim=-1).all():
                    break

        probs_text = torch.cat(probs_text_steps, dim=1)

        pred_tokens = tgt[:, 1:]
        eos_mask = (pred_tokens == self.eos_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.zeros((B,), dtype=torch.long, device=device)
        if has_eos.any():
            first_eos = eos_mask.float().argmax(dim=1)
        text_len_pred = torch.where(
            has_eos,
            first_eos,
            torch.full((B,), pred_tokens.size(1), device=device, dtype=torch.long),
        )
        text_len_pred = torch.clamp(text_len_pred, min=0)

        # Keep interface compatible: return one IDS sequence per final text position.
        # Positions beyond valid-head predicted valid length are represented as [].
        all_char_ids = []
        for b in range(B):
            n_final = min(int(text_len_pred[b].item()), max_chars)
            n_ids = min(int(valid_len_pred[b].item()), max_chars)
            cur = decoded_char_ids[b][:min(n_final, n_ids)]
            if n_final > len(cur):
                cur = cur + ([[]] * (n_final - len(cur)))
            all_char_ids.append(cur)

        return probs_text, all_char_ids, text_len_pred

    # -------------------- Forward --------------------
    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        return self.forward_test(x)


# python -m openrec.modeling.decoders.charwise_verify_style_decoder
if __name__ == "__main__":
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)
    print("Testing CharWiseVerifyStyleDecoder")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d_model = 256
    nhead = 4
    bs = 2
    max_text_len = 15
    ids_target_max_len = 15
    feat_seq_len = 32

    text_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_dict.txt")
    ids_vocab_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/minimal_ids_dict.txt")
    char_to_ids_path = os.path.join(project_root, "tools/utils/dict/visual_c3_ids/char_to_ids.txt")

    for p in [text_vocab_path, ids_vocab_path, char_to_ids_path]:
        assert os.path.exists(p), f"Not found: {p}"

    try:
        decoder = CharWiseVerifyStyleDecoder(
            in_channels=d_model,
            text_vocab_path=text_vocab_path,
            ids_vocab_path=ids_vocab_path,
            char_to_ids_path=char_to_ids_path,
            nhead=nhead,
            num_decoder_layers=1,
            dim_feedforward=512,
            max_text_length=max_text_len,
            cross_coverage=False,
            self_coverage=False,
            use_space_char=True,
            constrained_ctc_decode=True,
            ctc_frames_per_char=32,
            num_ctc_encoder_layers=1,
            use_valid_head=True,
        ).to(device)
        print(f"[OK] Decoder initialized. text_vocab={decoder.text_vocab_size}, ids_vocab={decoder.ids_vocab_size}")
        print(f"     CTC frames per char: {decoder.ctc_frames}")
        print(f"     Params: {sum(p.numel() for p in decoder.parameters()):,}")

        # ---- TRAIN forward ----
        decoder.train()
        x = torch.randn(bs, feat_seq_len, d_model, device=device)
        text_lens_val = [3, 5]
        ids_label_len = ids_target_max_len + 2

        text_labels = torch.randint(4, decoder.text_vocab_size, (bs, 2 + max_text_len), device=device)
        text_labels[:, 0] = 1
        for b in range(bs):
            text_labels[b, 1 + text_lens_val[b]] = 2
            text_labels[b, 2 + text_lens_val[b]:] = 0
        text_lengths = torch.tensor(text_lens_val, dtype=torch.long, device=device)

        per_char_ids_labels = torch.randint(4, decoder.ids_vocab_size, (bs, max_text_len, ids_label_len), device=device)
        per_char_ids_lengths = torch.zeros(bs, max_text_len, dtype=torch.long, device=device)
        for b in range(bs):
            for ci in range(text_lens_val[b]):
                ids_len = 3
                per_char_ids_labels[b, ci, 0] = 1
                per_char_ids_labels[b, ci, 1 + ids_len] = 2
                per_char_ids_labels[b, ci, 2 + ids_len:] = 0
                per_char_ids_lengths[b, ci] = ids_len
            per_char_ids_labels[b, text_lens_val[b]:] = 0

        data = [text_labels, text_lengths, per_char_ids_labels, per_char_ids_lengths]

        print("\nRunning forward_train...")
        logits_text, ids_output, char_feat, grammar_penalty, max_text = decoder(x, data)
        ids_ctc_loss, all_char_ids_train, valid_loss = ids_output

        print(f"  logits_text shape: {logits_text.shape}")
        print(f"  ids_ctc_loss:      {ids_ctc_loss.item():.4f}")
        print(f"  grammar_penalty:   {grammar_penalty.item():.4f}")
        print(f"  valid_loss:        {valid_loss.item():.4f}")
        print(f"  #batches decoded:  {len(all_char_ids_train)}")
        print(f"  char_feat shape:   {char_feat.shape}")
        print(f"  max_text:          {max_text}")

        assert logits_text.shape[0] == bs
        assert logits_text.shape[2] == decoder.text_vocab_size
        assert ids_ctc_loss.dim() == 0
        assert len(all_char_ids_train) == bs
        for b in range(bs):
            assert len(all_char_ids_train[b]) == text_lens_val[b]
            print(f"  batch {b}: {text_lens_val[b]} chars, IDS lengths = "
                  f"{[len(ids) for ids in all_char_ids_train[b]]}")
        print("  [OK] Train verified!")

        loss = logits_text.sum() + ids_ctc_loss + grammar_penalty + valid_loss
        loss.backward()
        grad_ok = all(p.grad is not None for p in decoder.parameters() if p.requires_grad)
        print(f"  [OK] Gradient flow: {grad_ok}")
        decoder.zero_grad()

        # ---- EVAL forward ----
        decoder.eval()
        print("\nRunning forward_test...")
        with torch.no_grad():
            probs_text, all_char_ids, text_len_pred = decoder(x)

        print(f"  probs_text shape:  {probs_text.shape}")
        print(f"  text_len_pred:     {text_len_pred.tolist()}")
        for b in range(bs):
            n_chars = int(text_len_pred[b].item())
            assert len(all_char_ids[b]) == n_chars
            print(f"  batch {b}: {n_chars} chars, IDS lengths = {[len(ids) for ids in all_char_ids[b]]}")
        print("  [OK] Eval verified!")

        print("\n" + "=" * 60)
        print("[PASS] All CharWiseVerifyStyleDecoder tests passed!")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {e}")