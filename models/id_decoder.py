# Copyright (c) RuopengGao. All Rights Reserved.
# About: Decoding the ID infos of current frame's objects.
import einops
import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint
from models.ffn import FFN
from models.utils import get_clones
from utils.utils import labels_to_one_hot, pos_to_pos_embed


class IDDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            dim_feedforward: int,
            n_heads: int,
            dropout: float,
            n_layers: int,
            device: str,
            num_id_vocabulary: int,
            max_temporal_length: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.num_id_vocabulary = num_id_vocabulary
        self.max_temporal_length = max_temporal_length

        # Cat(Features, ID Embeds) -> Hidden Dim
        self.attn_dim = 2 * self.hidden_dim
        self.attn_heads = 2 * self.n_heads
        self.ffn_dim = 2 * self.dim_feedforward

        # ID Word Embedding:
        self.word_to_embed = nn.Linear(in_features=self.num_id_vocabulary + 1, out_features=self.hidden_dim, bias=False)
        self.embed_to_word = nn.Linear(in_features=self.hidden_dim, out_features=self.num_id_vocabulary + 1, bias=False)

        # Build the relative temporal positional embedding:
        self.related_temporal_embeds = nn.Parameter(
            torch.zeros((self.n_layers, self.max_temporal_length, self.attn_heads), dtype=torch.float,
                        device=self.device)
        )
        t_idxs = torch.arange(self.max_temporal_length)
        t_in_dim0, t_in_dim1 = torch.meshgrid([t_idxs, t_idxs])
        self.related_temporal_pe_idx_map = (t_in_dim0 - t_in_dim1).to(torch.long).to(self.device)

        # ID Decoder networks:
        decoder_layer = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.attn_heads,
            dropout=dropout,
            batch_first=True,
            add_zero_attn=True,
        )
        dropout_layer = nn.Dropout(self.dropout)
        ffn_layer = FFN(
            d_model=self.attn_dim,
            d_ffn=self.ffn_dim,
            dropout=self.dropout,
            activation="GELU",
        )
        norm_layer = nn.LayerNorm(self.attn_dim)
        # Repeat the layers:
        self.decoder_layers = get_clones(decoder_layer, self.n_layers)
        self.dropout_layers = get_clones(dropout_layer, self.n_layers)
        self.ffn_layers = get_clones(ffn_layer, self.n_layers)
        self.norm_layers = get_clones(norm_layer, self.n_layers)
        # The self-attn layers in ID Decoder:
        self_attn_layer = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.attn_heads,
            dropout=self.dropout,
            batch_first=True,
            add_zero_attn=True,
        )
        # We remove the first self-attn layer in ID Decoder,
        # I think it is not necessary to use self-attn in the first layer.
        self.self_attn_layers = get_clones(self_attn_layer, self.n_layers - 1)
        self.self_norm_layers = get_clones(norm_layer, self.n_layers - 1)
        self.self_drop_layers = get_clones(dropout_layer, self.n_layers - 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Re-init the relative temporal embedding:
        self.related_temporal_embeds.data = self.related_temporal_embeds.data * 0

        return

    def forward(
            self,
            trajectory_feature_embeds, trajectory_masks,
            trajectory_ids, trajectory_times,
            unknown_features, unknown_times, unknown_ids, unknown_masks,
            use_checkpoint,
    ):
        trajectory_ids_one_hot = labels_to_one_hot(
            labels=einops.rearrange(trajectory_ids, "n t -> (n t)"),
            class_num=self.num_id_vocabulary + 1,
            device=trajectory_ids.device
        ).reshape((trajectory_ids.shape[0], trajectory_ids.shape[1], -1))
        unknown_ids_one_hot = labels_to_one_hot(
            labels=einops.rearrange(unknown_ids, "n t -> (n t)"),
            class_num=self.num_id_vocabulary + 1,
            device=trajectory_ids.device
        ).reshape((unknown_ids.shape[0], unknown_ids.shape[1], -1))
        trajectory_id_embeds = self.word_to_embed(trajectory_ids_one_hot)
        unknown_id_embeds = self.word_to_embed(unknown_ids_one_hot)

        # Get the relative temporal positional embedding for the current features:
        related_xy_flatten = torch.stack(
            torch.meshgrid([
                unknown_times.reshape((unknown_times.shape[0] * unknown_times.shape[1])),
                trajectory_times.reshape((trajectory_times.shape[0] * trajectory_times.shape[1]))
            ])
        ).permute(1, 2, 0)  # final to (T-1, T-1, 2)
        related_idx_flatten = self.related_temporal_pe_idx_map[
            related_xy_flatten[..., 0], related_xy_flatten[..., 1]
        ]

        # Combine the features and ID Embeds, get 2C-dim embeds:
        trajectory_embeds = torch.cat([
            trajectory_feature_embeds, trajectory_id_embeds
        ], dim=-1)
        unknown_embeds = torch.cat([
            unknown_features, unknown_id_embeds
        ], dim=-1)

        # Some metadata：
        N_trajectory, T_trajectory = trajectory_ids.shape
        N_unknown, T_unknown = unknown_ids.shape

        # Get the decoder mask, which is a causal mask:
        if self.training:
            assert T_trajectory == T_unknown and N_trajectory == N_unknown
            decoder_mask = nn.Transformer.generate_square_subsequent_mask(
                sz=T_trajectory, device=trajectory_embeds.device
            )
            decoder_mask = ~(torch.exp(decoder_mask).to(torch.bool))
            decoder_mask = decoder_mask[None, :, None, :].repeat((N_trajectory, 1, N_trajectory, 1))
            decoder_mask = einops.rearrange(decoder_mask, "a b c d -> (a b) (c d)")
        else:
            assert T_unknown == 1
            decoder_mask = None

        # Flatten the embeds and masks:
        trajectory_embeds_flatten = einops.rearrange(trajectory_embeds, "n t c -> (n t) c")
        unknown_embeds_flatten = einops.rearrange(unknown_embeds, "n t c -> (n t) c")
        trajectory_masks_flatten = einops.rearrange(trajectory_masks, "n t -> (n t)")

        def process_a_layer(
                _layer: int,    # which layer to process
                _unknown_embeds_flatten,
                _trajectory_embeds_flatten,
                _trajectory_masks_flatten,
                _decoder_mask,
                _related_idxs_flatten=None,
        ):
            if _layer > 0:          # we do not use self-attn at the first layer
                _self_unknown_embeds = einops.rearrange(_unknown_embeds_flatten, "(n t) c -> t n c", n=N_unknown, t=T_unknown)
                _self_unknown_masks = einops.rearrange(unknown_masks, "n t -> t n")
                _self_tgts, _ = self.self_attn_layers[_layer - 1](
                    query=_self_unknown_embeds, key=_self_unknown_embeds, value=_self_unknown_embeds,
                    key_padding_mask=_self_unknown_masks,
                )
                _self_unknown_embeds = self.self_norm_layers[_layer - 1](
                    _self_unknown_embeds + self.dropout_layers[_layer - 1](_self_tgts)
                )
                _unknown_embeds_flatten = einops.rearrange(_self_unknown_embeds, "t n c -> (n t) c")
                pass

            if _related_idxs_flatten is not None:   # turn the decoder mask from bool to float
                related_mask = self.related_temporal_embeds[_layer][_related_idxs_flatten]
                if _decoder_mask is not None:
                    related_mask = torch.masked_fill(
                        related_mask,
                        mask=_decoder_mask[..., None].repeat(1, 1, self.attn_heads),
                        value=float("-inf")
                    )
                related_mask = einops.rearrange(related_mask, "l s heads -> heads l s")[None, ...]
                # related_mask = related_mask.repeat(_unknown_embeds_flatten.shape[0], 1, 1, 1)
                # Do not need to repeat
                related_mask = einops.rearrange(related_mask, "b heads l s -> (b heads) l s")
                _decoder_mask = related_mask

            # Fix key_padding_mask type in PyTorch 2.1:
            # TODO: to support both PyTorch >= 2.0 and PyTorch <= 1.13
            key_padding_mask = _trajectory_masks_flatten[None, ...]
            # true to -inf, false to 0
            key_padding_mask = torch.masked_fill(key_padding_mask.to(torch.float), mask=key_padding_mask, value=float("-inf"))
            _unknown_tgts_flatten, _ = self.decoder_layers[_layer](
                query=_unknown_embeds_flatten[None, ...], key=_trajectory_embeds_flatten[None, ...], value=_trajectory_embeds_flatten[None, ...],
                key_padding_mask=key_padding_mask, attn_mask=_decoder_mask
            )
            _unknown_tgts_flatten = _unknown_tgts_flatten[0]

            _unknown_embeds_flatten = self.norm_layers[_layer](
                _unknown_embeds_flatten + self.dropout_layers[_layer](_unknown_tgts_flatten)
            )
            _unknown_embeds_flatten = self.ffn_layers[_layer](_unknown_embeds_flatten)
            return _unknown_embeds_flatten

        for _layer in range(self.n_layers):
            if use_checkpoint:
                unknown_embeds_flatten = checkpoint(
                    process_a_layer,
                    use_reentrant=False,
                    _layer=_layer,
                    _unknown_embeds_flatten=unknown_embeds_flatten,
                    _trajectory_embeds_flatten=trajectory_embeds_flatten,
                    _trajectory_masks_flatten=trajectory_masks_flatten,
                    _decoder_mask=decoder_mask,
                    _related_idxs_flatten=related_idx_flatten,
                )
            else:
                unknown_embeds_flatten = process_a_layer(
                    _layer=_layer,
                    _unknown_embeds_flatten=unknown_embeds_flatten,
                    _trajectory_embeds_flatten=trajectory_embeds_flatten,
                    _trajectory_masks_flatten=trajectory_masks_flatten,
                    _decoder_mask=decoder_mask,
                    _related_idxs_flatten=related_idx_flatten,
                )

        # Get the final ID words (embeddings):
        _, unknown_id_embeds_flatten = torch.chunk(unknown_embeds_flatten, chunks=2, dim=-1)

        # Get the legal ID word masks, and remove the illegal ID words:
        legal_id_embeds = unknown_id_embeds_flatten[~einops.rearrange(unknown_masks, "n t -> (n t)")]

        legal_id_words = self.embed_to_word(legal_id_embeds)

        return legal_id_words
