from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordPosEnc(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 500, temperature: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))
        inv_freq = torch.einsum("i, j -> i j", position, div_term)
        pe[:, 0::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.size()
        return x + self.pe[:L, :][None, :, :]


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage
        in_chs = (nhead if (cross_coverage ^ self_coverage) else 2 * nhead)
        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(nhead)

    def forward(self, prev_attn: torch.Tensor, key_padding_mask: Optional[torch.Tensor], h: int, curr_attn: torch.Tensor) -> torch.Tensor:
        """Compute coverage refinement term R from historical attention.

        Args:
            prev_attn: [B, n, T, L] previous-layer refined attn (or current as fallback)
            key_padding_mask: [B, L] bool, True for pad positions
            h: encoder feature map height
            curr_attn: [B, n, T, L] current-layer attn (softmax)
        Returns:
            cov term R with shape [B, n, T, L]
        """
        B, n, T, L = curr_attn.shape
        assert L % h == 0, f"Coverage reshape mismatch: L={L} not divisible by h={h}"
        w = L // h
        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)  # [B, C, T, L]

        # mask pad positions to avoid leaking coverage on padded spatial locations
        if key_padding_mask is not None:
            mask = key_padding_mask.bool()[:, None, None, :]  # [B,1,1,L]
            attns = attns.masked_fill(mask, 0.0)

        # cumulative coverage up to t-1
        cov = attns.cumsum(dim=2) - attns  # [B, C, T, L]
        cov = cov.permute(0, 2, 1, 3).contiguous().view(B * T, attns.size(1), h, w)  # [B*T, C, H, W]
        cov = self.proj(self.act(self.conv(cov)))
        cov = self.bn(cov)
        cov = cov.view(B, T, self.nhead, h, w).permute(0, 2, 1, 3, 4).contiguous()
        cov = cov.view(B, self.nhead, T, L)  # [B, n, T, L]
        return cov


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        arm: Optional[AttentionRefinementModule] = None,
        prev_attn: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention with optional ARM coverage.

        Args:
            query: [Tq, B, C]
            key/value: [Sk, B, C]
            prev_attn: [B, n, Tq, Sk] from previous decoder layer (refined), or None
            height: encoder feature map height (for reshape inside ARM)
            key_padding_mask: [B, Sk] bool
        Returns:
            attn_output: [Tq, B, C]
            attn_weights: refined attn [B, n, Tq, Sk]
        """
        Tq, B, C = query.shape
        Sk = key.shape[0]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()

        q = self.q_proj(query)  # [Tq, B, C]
        k = self.k_proj(key)
        v = self.v_proj(value)

        def shape(x):
            L = x.size(0)
            return x.view(L, B, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2).contiguous()  # [B, n, L, hd]

        q = shape(q)
        k = shape(k)
        v = shape(v)

        # energy: [B, n, Tq, Sk]
        energy = torch.einsum('bnth,bnsh->bnts', q, k) * self.scale

        if key_padding_mask is not None:
            energy = energy.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                energy = energy.masked_fill(attn_mask, float('-inf'))
            else:
                energy = energy + attn_mask  # additive mask

        attn = F.softmax(energy, dim=-1)

        # coverage refinement
        if arm is not None and height is not None:
            prev = prev_attn if prev_attn is not None else attn
            cov = arm(prev, key_padding_mask, height, attn)
            cov = cov.to(dtype=energy.dtype)
            energy = energy - cov
            if key_padding_mask is not None:
                energy = energy.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
            attn = F.softmax(energy, dim=-1)

        attn = self.dropout(attn)
        out = torch.einsum('bnts,bnsh->bnth', attn, v)  # [B, n, Tq, hd]
        out = out.transpose(1, 2).contiguous().view(B, Tq, C).transpose(0, 1)  # [Tq, B, C]
        out = self.out_proj(out)
        return out, attn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(
        self,
        tgt,
        memory,
        arm=None,
        prev_cross_attn=None,
        height: Optional[int] = None,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        q = k = tgt
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, attn = self.cross_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            prev_attn=prev_cross_attn,
            height=height,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt, attn


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, arm: Optional[AttentionRefinementModule]):
        super().__init__()
        d_model = decoder_layer.linear2.out_features
        nhead = decoder_layer.self_attn.num_heads
        dim_ff = decoder_layer.linear1.out_features
        dropout = decoder_layer.dropout.p
        self.layers = nn.ModuleList([
            decoder_layer if i == 0 else type(decoder_layer)(d_model, nhead, dim_ff, dropout)
            for i in range(num_layers)
        ])
        self.arm = arm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, height: int, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attn=False):
        out = tgt
        prev_attn = None
        last_cross_attn = None
        for i, layer in enumerate(self.layers):
            out, attn = layer(
                out,
                memory,
                arm=self.arm,
                prev_cross_attn=prev_attn,
                height=height,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            last_cross_attn = attn
            if self.arm is not None:
                prev_attn = attn  # refined attention passed to next layer
        if return_attn:
            return self.norm(out), last_cross_attn
        return self.norm(out)


class TAMERDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 64,
        cross_coverage: bool = False,
        self_coverage: bool = False,
        max_label_length: int = 25,
        beam_size: int = 1,
        lambda_struct: float = 0.0,
        length_penalty_alpha: float = 0.0,
        early_stopping: bool = False,
    ):
        """TAMER 风格解码器（带可选 coverage）。

        参数说明：
        - in_channels: 输入特征维度（= 编码器 d_model）。
        - out_channels: 字典大小；与数据一致约定 <pad>=0, <sos>=1, <eos>=2。
        - nhead: 多头注意力头数。
        - num_decoder_layers: 解码器层数。
        - dim_feedforward: 前馈层维度。
        - dropout: 解码器各处 dropout 比例。
        - dc: coverage 卷积通道数（AttentionRefinementModule 的中间通道）。
        - cross_coverage: 是否启用交叉注意力的覆盖惩罚。
        - self_coverage: 是否在自注意力上也进行覆盖累积（与 cross_coverage 可并存）。
        - max_label_length: 推理时的最大字符数（不含 <eos>，内部会 +1）。
        - beam_size: beam 搜索宽度；为 1 时走贪心，大于 1 启用批量 beam（B*beam 并行）。
        - lambda_struct: 结构重排权重 λ；最终分数 = beam 序列分数 + λ × 结构分数。
        - length_penalty_alpha: 长度惩罚系数 α；beam 分数计算为 sum_logprobs / 长度^α（α=0 表示不惩罚）。
        - early_stopping: 是否在每个样本收集到 beam_size 个完整候选（到达 <eos>）后直接早停。
        输出：训练返回 logits [B, L, out_channels]；推理返回概率 [B, <=max_len+1, out_channels]。
        """
        super().__init__()
        self.out_channels = out_channels
        # 与预处理一致：<pad>=0, <sos>=1, <eos>=2
        self.ignore_index = 0
        self.bos_id = 1
        self.eos_id = 2
        self.max_len = max_label_length
        self.beam_size = beam_size
        self.lambda_struct = lambda_struct
        self.length_penalty_alpha = length_penalty_alpha
        self.early_stopping = early_stopping
        d_model = in_channels

        self.word_embed = nn.Sequential(nn.Embedding(out_channels, d_model), nn.LayerNorm(d_model))
        self.pos_enc = WordPosEnc(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_decoder_layers,
            arm=AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage) if (cross_coverage or self_coverage) else None,
        )
        self.proj = nn.Linear(d_model, out_channels)
        # 结构分支：TAM 风格，一层 TransformerEncoder + child/parent 投影 + ReLU + vs 打分
        self.struct_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=1,
        )
        self.child_proj = nn.Linear(d_model, d_model)
        self.parent_proj = nn.Linear(d_model, d_model)
        # 与官方一致，保留 bias：ReLU(child+parent) -> Linear(d_model->1, bias=True)
        self.vs = nn.Linear(d_model, 1, bias=True)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        m = torch.full((L, L), fill_value=1, dtype=torch.bool, device=device)
        m.triu_(1)
        return m

    def _prep_memory(self, x) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        # 支持 encoder 返回 (feat_2d, mask_2d)，其中 feat_2d:[B,H,W,C]，mask_2d:[B,H,W] (bool, True=pad)
        if isinstance(x, (tuple, list)):
            if len(x) == 2:
                feat2d, mask2d = x
                b, h, w, c = feat2d.shape
                mem = feat2d.view(b, h * w, c)
                mem_mask = mask2d.reshape(b, h * w).bool()
                return mem, h, mem_mask
            elif len(x) == 3:
                # 兼容 (seq, (h,w), mask_flat)
                mem, hw, mem_mask = x
                h, _ = hw
                return mem, h, mem_mask.bool() if mem_mask is not None else None
        # fallback: only seq given
        mem = x
        return mem, 2, None

    def _decode_hidden(self, memory, h, mem_mask, tgt_ids: torch.Tensor):
        """Teacher-forcing decode to obtain hidden states and logits for a given target sequence.

        Args:
            memory: [B, N, C]
            h: encoder height (for coverage, kept for consistency)
            mem_mask: [B, N] or None
            tgt_ids: [B, L] target token ids (include <sos> and <eos>, padded with <pad>)
        Returns:
            logits: [B, L, V]
            sim: [B, L, L]
        """
        pad_mask = (tgt_ids == self.ignore_index)
        L = tgt_ids.size(1)
        tgt_mask = self._causal_mask(L, device=memory.device)

        tgt_emb = self.word_embed(tgt_ids)
        tgt_emb = self.pos_enc(tgt_emb)
        tgt_emb = self.norm(tgt_emb)
        out = self.decoder(
            tgt=tgt_emb.transpose(0, 1),
            memory=memory.transpose(0, 1),
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=mem_mask,
        )
        out = out.transpose(0, 1)
        logits = self.proj(out)
        sim = self._struct_sim(out, pad_mask)
        return logits, sim

    def forward_train(self, x, labels: torch.Tensor, lengths: torch.Tensor):
        # 训练：teacher forcing，截到 batch 内最大长度
        memory, h, mem_mask = self._prep_memory(x)
        B, N, C = memory.shape
        max_len = int(lengths.max().item())
        tgt = labels[:, : 2 + max_len]
        L = tgt.size(1)
        tgt_mask = self._causal_mask(L, device=memory.device)  # 上三角 True 作自回归掩码
        pad_mask = (tgt == self.ignore_index)  # 忽略 PAD/BOS 以后的填充位

        tgt_emb = self.word_embed(tgt)
        tgt_emb = self.pos_enc(tgt_emb)
        tgt_emb = self.norm(tgt_emb)
        tgt_seq = tgt_emb.transpose(0, 1)  # [L, B, C]
        mem_seq = memory.transpose(0, 1)   # [N, B, C]
        out = self.decoder(
            tgt=tgt_seq,
            memory=mem_seq,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=mem_mask,
        )
        out = out.transpose(0, 1)  # [B, L, C]
        logits = self.proj(out)
        sim = self._struct_sim(out, pad_mask)  # [B, L, L]
        return logits, sim

    def forward_test(self, x):
        # 推理：若 beam_size>1 则 beam+结构重排；否则贪心
        memory, h, mem_mask = self._prep_memory(x)
        B, N, C = memory.shape
        if self.beam_size is None or self.beam_size <= 1:
            # 贪心
            num_steps = self.max_len + 1
            tgt = torch.full((B, 1), self.bos_id, dtype=torch.long, device=memory.device)
            logits = []
            for i in range(num_steps):
                L = tgt.size(1)
                tgt_mask = self._causal_mask(L, device=memory.device)
                pad_mask = (tgt == self.ignore_index)
                tgt_emb = self.word_embed(tgt)
                tgt_emb = self.pos_enc(tgt_emb)
                tgt_emb = self.norm(tgt_emb)
                out = self.decoder(
                    tgt=tgt_emb.transpose(0, 1),
                    memory=memory.transpose(0, 1),
                    height=h,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=pad_mask,
                    memory_key_padding_mask=mem_mask,
                )
                out = out.transpose(0, 1)
                p_i = self.proj(out[:, -1:, :])
                logits.append(F.softmax(p_i, -1))
                if i < self.max_len:
                    nxt = p_i.squeeze(1).argmax(-1)
                    tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
                    if (tgt == self.eos_id).any(dim=-1).all():
                        break
            return torch.cat(logits, dim=1)

        # Beam search + 结构重排（单向，批处理 B*beam 并行）
        device = memory.device
        batch_size = B
        beam_size = self.beam_size
        alpha = self.length_penalty_alpha

        class BeamHypotheses:
            def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
                self.length_penalty = length_penalty
                self.early_stopping = early_stopping
                self.num_beams = num_beams
                self.beams = []
                self.worst_score = 1e9

            def __len__(self):
                return len(self.beams)

            def add(self, hyp: torch.Tensor, sum_logprobs: float):
                score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
                if len(self) < self.num_beams or score > self.worst_score:
                    self.beams.append((score, hyp))
                    if len(self) > self.num_beams:
                        sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                        del self.beams[sorted_next_scores[0][1]]
                        self.worst_score = sorted_next_scores[1][0]
                    else:
                        self.worst_score = min(score, self.worst_score)

            def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
                if len(self) < self.num_beams:
                    return False
                if self.early_stopping:
                    return True
                cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)
                return self.worst_score >= cur_score

        class BeamSearchScorer:
            def __init__(self, batch_size: int, beam_size: int, alpha: float, early_stopping: bool, device, eos_id: int, ignore_index: int):
                self.batch_size = batch_size
                self.beam_size = beam_size
                self.alpha = alpha
                self.device = device
                self.eos_id = eos_id
                self.ignore_index = ignore_index
                self._beam_hyps = [BeamHypotheses(beam_size, alpha, early_stopping) for _ in range(batch_size)]
                self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)

            def is_done(self):
                return self._done.all()

            def process(self, input_ids, next_scores, next_tokens, next_indices):
                next_beam_scores = torch.zeros((self.batch_size, self.beam_size), device=self.device, dtype=next_scores.dtype)
                next_beam_tokens = torch.zeros((self.batch_size, self.beam_size), device=self.device, dtype=next_tokens.dtype)
                next_beam_indices = torch.zeros((self.batch_size, self.beam_size), device=self.device, dtype=next_indices.dtype)
                for batch_idx, beam_hyp in enumerate(self._beam_hyps):
                    if self._done[batch_idx]:
                        assert len(beam_hyp) >= self.beam_size
                        next_beam_scores[batch_idx] = 0
                        next_beam_tokens[batch_idx] = self.ignore_index
                        next_beam_indices[batch_idx] = batch_idx * self.beam_size
                        continue
                    beam_idx = 0
                    for beam_token_rank, (score, token, index) in enumerate(zip(next_scores[batch_idx], next_tokens[batch_idx], next_indices[batch_idx])):
                        batch_beam_idx = batch_idx * self.beam_size + index
                        is_eos = token.item() == self.eos_id
                        if is_eos:
                            if beam_token_rank >= self.beam_size:
                                continue
                            beam_hyp.add(input_ids[batch_beam_idx].clone(), score.item())
                        else:
                            next_beam_scores[batch_idx, beam_idx] = score
                            next_beam_tokens[batch_idx, beam_idx] = token
                            next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                            beam_idx += 1
                        if beam_idx == self.beam_size:
                            break
                    assert beam_idx == self.beam_size
                    self._done[batch_idx] = beam_hyp.is_done(best_sum_logprobs=next_beam_scores[batch_idx].max().item(), cur_len=input_ids.shape[-1])
                return next_beam_scores.view(-1), next_beam_tokens.view(-1), next_beam_indices.view(-1)

            def finalize(self, input_ids, final_scores):
                for batch_idx, beam_hyp in enumerate(self._beam_hyps):
                    if self._done[batch_idx]:
                        continue
                    for beam_id in range(self.beam_size):
                        batch_beam_idx = batch_idx * self.beam_size + beam_id
                        beam_hyp.add(input_ids[batch_beam_idx], final_scores[batch_beam_idx].item())
                all_hyps = []
                scores = torch.zeros(self.batch_size * self.beam_size, device=self.device)
                for beam_hyp in self._beam_hyps:
                    for s, seq in beam_hyp.beams:
                        scores[len(all_hyps)] = s
                        all_hyps.append(seq[1:])  # drop BOS
                return all_hyps, scores

        beam_scorer = BeamSearchScorer(batch_size, beam_size, alpha, self.early_stopping, device, self.eos_id, self.ignore_index)
        input_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(batch_size, device=device)
        mem_cur = memory
        mem_mask_cur = mem_mask
        cur_len = input_ids.size(1)
        while cur_len <= self.max_len and not beam_scorer.is_done():
            tgt_mask = self._causal_mask(cur_len, device=device)
            pad_mask = (input_ids == self.ignore_index)
            tgt_emb = self.word_embed(input_ids)
            tgt_emb = self.pos_enc(tgt_emb)
            tgt_emb = self.norm(tgt_emb)
            out = self.decoder(
                tgt=tgt_emb.transpose(0, 1),
                memory=mem_cur.transpose(0, 1),
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=mem_mask_cur,
            ).transpose(0, 1)
            logits_step = self.proj(out)[:, -1, :]  # [B*beam, V]
            logp = F.log_softmax(logits_step, dim=-1)
            # add beam scores
            logp = logp + beam_scores.unsqueeze(1)
            # reshape to [B, beam*V]
            current_beam = logp.size(0) // batch_size
            logp = logp.view(batch_size, current_beam * self.out_channels)
            next_scores, next_tokens = torch.topk(logp, k=min(logp.size(1), 2 * beam_size), dim=1)
            next_indices = next_tokens // self.out_channels
            next_tokens = next_tokens % self.out_channels
            if cur_len == 1:
                # expand input and memory for beam_size
                input_ids = input_ids.repeat_interleave(beam_size, dim=0)
                if mem_cur is not None:
                    mem_cur = mem_cur.repeat_interleave(beam_size, dim=0)
                if mem_mask_cur is not None:
                    mem_mask_cur = mem_mask_cur.repeat_interleave(beam_size, dim=0)
                beam_scores = beam_scores.repeat_interleave(beam_size, dim=0)
                current_beam = beam_size
            beam_scores, beam_next_tokens, beam_indices = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )
            input_ids = torch.cat([input_ids[beam_indices], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len += 1

        hyps, scores = beam_scorer.finalize(input_ids, beam_scores)
        # hyps: list len B*beam, scores tensor B*beam
        results = []
        hyps_per_batch = [hyps[i * beam_size:(i + 1) * beam_size] for i in range(batch_size)]
        scores_per_batch = scores.view(batch_size, beam_size)
        pad_len = self.max_len + 2
        for b in range(batch_size):
            mem_b = memory[b:b + 1]
            mem_mask_b = mem_mask[b:b + 1] if mem_mask is not None else None
            seq_scores = scores_per_batch[b].tolist()
            selected_tokens = []
            for hyp in hyps_per_batch[b]:
                tokens = hyp.tolist()
                tokens = [self.bos_id] + tokens
                if not tokens or tokens[-1] != self.eos_id:
                    tokens = tokens + [self.eos_id]
                selected_tokens.append(tokens)
            # pad
            tgt_batch_sel = torch.full((beam_size, pad_len), self.ignore_index, device=device, dtype=torch.long)
            for i, tokens in enumerate(selected_tokens):
                tokens = tokens[: pad_len]
                if len(tokens) < pad_len:
                    tokens = tokens + [self.ignore_index] * (pad_len - len(tokens))
                tgt_batch_sel[i] = torch.tensor(tokens, device=device, dtype=torch.long)
            logits_b, sim_b = self._decode_hidden(
                mem_b.expand(beam_size, -1, -1),
                h,
                mem_mask_b.expand(beam_size, -1) if mem_mask_b is not None else None,
                tgt_batch_sel,
            )
            log_sim = F.log_softmax(sim_b, dim=-1)
            pad_mask_sel = (tgt_batch_sel == self.ignore_index)
            valid_rows = ~pad_mask_sel
            valid_rows[:, 0] = False
            struct_scores = []
            for i in range(beam_size):
                vals = []
                for r in range(log_sim.size(1)):
                    if not valid_rows[i, r]:
                        continue
                    vals.append(float(log_sim[i, r].max().item()))
                struct_scores.append(sum(vals) / max(1, len(vals)))
            comb = [s + self.lambda_struct * t for s, t in zip(seq_scores, struct_scores)]
            best_idx = int(torch.tensor(comb).argmax().item()) if len(comb) > 0 else 0
            logits = F.softmax(logits_b[best_idx: best_idx + 1], dim=-1)
            results.append(logits)

        return torch.cat(results, dim=0)

    def forward(self, x, data=None):
        if self.training:
            # print("data:", data)
            # data: [label, length, ...]
            labels = data[0]
            lengths = data[1]
            return self.forward_train(x, labels, lengths)
        else:
            return self.forward_test(x)

    def _struct_sim(self, feat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        结构相似度矩阵：行 i 对应 child i，列 j 对应 parent j，使用点积相似度。
        - feat: [B, L, C]
        - pad_mask: [B, L]，True 表示 PAD/忽略，不参与父节点候选。
        返回：sim_logits [B, L, L]（未 softmax，便于外部用 CE + mask）。
        """
        # 先过一层 TransformerEncoder，获得上下文增强的 token 表示
        feat = self.struct_enc(feat, src_key_padding_mask=pad_mask)
        child = self.child_proj(feat)   # [B, L, C]
        parent = self.parent_proj(feat) # [B, L, C]
        # M_{i,j} = ReLU(child_i + parent_j)
        M = F.relu(child.unsqueeze(2) + parent.unsqueeze(1))  # [B, L, L, C]
        sim = self.vs(M).squeeze(-1)  # [B, L, L]
        # 屏蔽 BOS 列 + padding 列（不可作为 parent），child 行的忽略交由 loss 处理
        if pad_mask is not None:
            pm = pad_mask.clone()
            pm[:, 0] = True  # BOS 不作为 parent 候选
            sim = sim.masked_fill(pm[:, None, :], float('-inf'))
        return sim
