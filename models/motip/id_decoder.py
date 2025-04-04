# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
from typing import Tuple
from torch.utils.checkpoint import checkpoint

from models.misc import _get_clones, label_to_one_hot
from models.ffn import FFN


class IDDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            id_dim: int,
            ffn_dim_ratio: int,
            num_layers: int,
            head_dim: int,
            num_id_vocabulary: int,
            rel_pe_length: int,
            use_aux_loss: bool,
            use_shared_aux_head: bool,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.n_heads = (self.feature_dim + self.id_dim) // self.head_dim
        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length

        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head

        self.word_to_embed = nn.Linear(self.num_id_vocabulary + 1, self.id_dim, bias=False)
        embed_to_word = nn.Linear(self.id_dim, self.num_id_vocabulary + 1, bias=False)

        if self.use_aux_loss and not self.use_shared_aux_head:
            self.embed_to_word_layers = _get_clones(embed_to_word, self.num_layers)
        else:
            self.embed_to_word_layers = nn.ModuleList([embed_to_word for _ in range(self.num_layers)])
        pass

        # Related Position Embeddings:
        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((self.num_layers, self.rel_pe_length, self.n_heads), dtype=torch.float32)
        )
        # Prepare others for rel pe:
        t_idxs = torch.arange(self.rel_pe_length, dtype=torch.int64)
        curr_t_idxs, traj_t_idxs = torch.meshgrid([t_idxs, t_idxs])
        self.rel_pos_map = (curr_t_idxs - traj_t_idxs)      # [curr_t_idx, traj_t_idx] -> rel_pos, like [1, 0] = 1
        pass

        self_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        self_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim + self.id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)
        ffn = FFN(
            d_model=self.feature_dim + self.id_dim,
            d_ffn=(self.feature_dim + self.id_dim) * self.ffn_dim_ratio,
            activation=nn.GELU(),
        )
        ffn_norm = nn.LayerNorm(self.feature_dim + self.id_dim)

        self.self_attn_layers = _get_clones(self_attn, self.num_layers - 1)
        self.self_attn_norm_layers = _get_clones(self_attn_norm, self.num_layers - 1)
        self.cross_attn_layers = _get_clones(cross_attn, self.num_layers)
        self.cross_attn_norm_layers = _get_clones(cross_attn_norm, self.num_layers)
        self.ffn_layers = _get_clones(ffn, self.num_layers)
        self.ffn_norm_layers = _get_clones(ffn_norm, self.num_layers)

        # Init parameters:
        for n, p in self.named_parameters():
            if p.dim() > 1 and "rel_pos_embeds" not in n:
                nn.init.xavier_uniform_(p)

        pass

    def forward(self, seq_info, use_decoder_checkpoint):
        trajectory_features = seq_info["trajectory_features"]
        unknown_features = seq_info["unknown_features"]
        trajectory_id_labels = seq_info["trajectory_id_labels"]
        unknown_id_labels = seq_info["unknown_id_labels"] if "unknown_id_labels" in seq_info else None
        trajectory_times = seq_info["trajectory_times"]
        unknown_times = seq_info["unknown_times"]
        trajectory_masks = seq_info["trajectory_masks"]
        unknown_masks = seq_info["unknown_masks"]
        _B, _G, _T, _N, _ = trajectory_features.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_features.shape

        trajectory_id_embeds = self.id_label_to_embed(id_labels=trajectory_id_labels)
        unknown_id_embeds = self.generate_empty_id_embed(unknown_features=unknown_features)

        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)

        # Prepare some common variables:
        self_attn_key_padding_mask = einops.rearrange(unknown_masks, "b g t n -> (b g t) n").contiguous()
        cross_attn_key_padding_mask = einops.rearrange(trajectory_masks, "b g t n -> (b g) (t n)").contiguous()
        _trajectory_times_flatten = einops.rearrange(trajectory_times, "b g t n -> (b g) (t n)")
        _unknown_times_flatten = einops.rearrange(unknown_times, "b g t n -> (b g) (t n)")
        cross_attn_mask = _trajectory_times_flatten[:, None, :] >= _unknown_times_flatten[:, :, None]
        cross_attn_mask = einops.repeat(cross_attn_mask, "bg tn1 tn2 -> (bg n_heads) tn1 tn2", n_heads=self.n_heads).contiguous()
        # Prepare for rel PE:
        self.rel_pos_map = self.rel_pos_map.to(trajectory_features.device)
        rel_pe_idx_pairs = torch.stack([
            torch.stack(
                torch.meshgrid([_unknown_times_flatten[_], _trajectory_times_flatten[_]]), dim=-1
            )
            for _ in range(len(_trajectory_times_flatten))
        ], dim=0)       # (B*G, T*N of curr, T*N of traj, 2)
        rel_pe_idx_pairs = rel_pe_idx_pairs.to(trajectory_features.device)
        rel_pe_idxs = self.rel_pos_map[rel_pe_idx_pairs[..., 0], rel_pe_idx_pairs[..., 1]]      # (B*G, T_curr, T_traj)
        pass
        # Change Cross-Attn key_padding_mask and attn_mask to float:
        cross_attn_key_padding_mask = torch.masked_fill(
            cross_attn_key_padding_mask.float(),
            mask=cross_attn_key_padding_mask,
            value=float("-inf"),
        ).to(self.dtype)
        cross_attn_mask = torch.masked_fill(
            cross_attn_mask.float(),
            mask=cross_attn_mask,
            value=float("-inf"),
        ).to(self.dtype)
        pass

        all_unknown_id_logits = None
        all_unknown_id_labels = None
        all_unknown_id_masks = None

        for layer in range(self.num_layers):
            # Predict ID logits:
            if use_decoder_checkpoint:
                unknown_embeds = checkpoint(
                    self._forward_a_layer,
                    layer,
                    unknown_embeds, trajectory_embeds,
                    self_attn_key_padding_mask, cross_attn_key_padding_mask,
                    cross_attn_mask, rel_pe_idxs,
                    use_reentrant=False,
                )
            else:
                unknown_embeds = self._forward_a_layer(
                    layer=layer,
                    unknown_embeds=unknown_embeds,
                    trajectory_embeds=trajectory_embeds,
                    self_attn_key_padding_mask=self_attn_key_padding_mask,
                    cross_attn_key_padding_mask=cross_attn_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pe_idx=rel_pe_idxs,
                )

            _unknown_id_logits = self.embed_to_word_layers[layer](unknown_embeds[..., -self.id_dim:])
            _unknown_id_masks = unknown_masks.clone()
            _unknown_id_labels = None if not self.training else unknown_id_labels
            if all_unknown_id_logits is None:
                all_unknown_id_logits = _unknown_id_logits
                all_unknown_id_labels = _unknown_id_labels
                all_unknown_id_masks = _unknown_id_masks
            else:
                all_unknown_id_logits = torch.cat([all_unknown_id_logits, _unknown_id_logits], dim=0)
                all_unknown_id_labels = torch.cat([all_unknown_id_labels, _unknown_id_labels], dim=0) if _unknown_id_labels is not None else None
                all_unknown_id_masks = torch.cat([all_unknown_id_masks, _unknown_id_masks], dim=0)

        if self.training and self.use_aux_loss:
            return all_unknown_id_logits, all_unknown_id_labels, all_unknown_id_masks
        else:
            return _unknown_id_logits, _unknown_id_labels, _unknown_id_masks

    def _forward_a_layer(
            self,
            layer: int,
            unknown_embeds: torch.Tensor,
            trajectory_embeds: torch.Tensor,
            self_attn_key_padding_mask: torch.Tensor,
            cross_attn_key_padding_mask: torch.Tensor,
            cross_attn_mask: torch.Tensor,
            rel_pe_idx: torch.Tensor,
    ):
        _B, _G, _T, _N, _ = trajectory_embeds.shape
        _curr_B, _curr_G, _curr_T, _curr_N, _ = unknown_embeds.shape
        if layer > 0:   # use self-attention to transfer information between unknown features (same time step)
            self_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g t) n c").contiguous()
            self_out, _ = self.self_attn_layers[layer - 1](
                query=self_unknown_embeds, key=self_unknown_embeds, value=self_unknown_embeds,
                key_padding_mask=self_attn_key_padding_mask,
            )
            self_out = self_unknown_embeds + self_out
            self_out = self.self_attn_norm_layers[layer - 1](self_out)
            unknown_embeds = einops.rearrange(self_out, "(b g t) n c -> b g t n c", b=_B, g=_G, t=_curr_T)

        # Cross-attention for in-context decoding:
        cross_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        cross_trajectory_embeds = einops.rearrange(trajectory_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        # Prepare attn_mask:
        rel_pe_mask = self.rel_pos_embeds[layer][rel_pe_idx]
        cross_attn_mask_with_rel_pe = cross_attn_mask + einops.rearrange(rel_pe_mask, "bg l1 l2 n -> (bg n) l1 l2")
        # Apply cross-attention:
        cross_out, _ = self.cross_attn_layers[layer](
            query=cross_unknown_embeds, key=cross_trajectory_embeds, value=cross_trajectory_embeds,
            key_padding_mask=cross_attn_key_padding_mask,
            attn_mask=cross_attn_mask_with_rel_pe,
        )
        cross_out = cross_unknown_embeds + cross_out
        cross_out = self.cross_attn_norm_layers[layer](cross_out)
        # Feed-forward network:
        cross_out = cross_out + self.ffn_layers[layer](cross_out)
        cross_out = self.ffn_norm_layers[layer](cross_out)
        # Re-shape back to original shape:
        unknown_embeds = einops.rearrange(cross_out, "(b g) (t n) c -> b g t n c", b=_B, g=_G, t=_curr_T)

        return unknown_embeds

    def id_label_to_embed(self, id_labels):
        id_words = label_to_one_hot(id_labels, self.num_id_vocabulary + 1, dtype=self.dtype)
        id_embeds = self.word_to_embed(id_words)
        return id_embeds

    def generate_empty_id_embed(self, unknown_features):
        _shape = unknown_features.shape[:-1]
        empty_id_labels = self.num_id_vocabulary * torch.ones(_shape, dtype=torch.int64, device=unknown_features.device)
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds

    def shuffle(self):
        shuffle_index = torch.randperm(self.num_id_vocabulary, device=self.word_to_embed.weight.device)
        shuffle_index = torch.cat([shuffle_index, torch.tensor([self.num_id_vocabulary], device=self.word_to_embed.weight.device)])
        self.word_to_embed.weight.data = self.word_to_embed.weight.data[:, shuffle_index]
        self.embed_to_word.weight.data = self.embed_to_word.weight.data[shuffle_index, :]
        pass

    @property
    def dtype(self):
        return self.word_to_embed.weight.dtype