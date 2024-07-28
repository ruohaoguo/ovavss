import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP


class DecoupledTemporalInstanceResampler(nn.Module):

    def __init__(self, hidden_dim=256, feed_dim=2048, nqueries=100, nheads=8, nlayers=6):
        super().__init__()
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = nlayers

        self.long_aggregate_layers = nn.ModuleList()
        self.short_aggregate_layers = nn.ModuleList()
        self.aggregate_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        self.tgt_ca_layers = nn.ModuleList()
        self.tgt_sa_layers = nn.ModuleList()
        self.tgt_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.long_aggregate_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.aggregate_norms.append(nn.LayerNorm(hidden_dim))

            self.ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feed_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.tgt_sa_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.tgt_ca_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.tgt_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feed_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decode_norm = nn.LayerNorm(hidden_dim)

        self.nqueries = nqueries
        # learnable query features
        self.query_emb = nn.Embedding(nqueries, hidden_dim)
        # learnable query p.e.
        self.query_pos = nn.Embedding(nqueries, hidden_dim)

        # output FFNs
        self.attn_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.adapter = None
        self.text_feats = None

    def forward(self, frame_embeds, mask_feats, attn_feats, adapter=None, clip_bk_feats=None, text_feats=None):
        self.adapter = adapter
        self.text_feats = text_feats

        predictions_class = []
        predictions_mask = []

        bs, t, q = frame_embeds.shape[:3]

        # QxNxC
        tgt = self.query_emb.weight.unsqueeze(1).repeat(1, bs * t, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs * t, 1)

        logits = self.forward_class_predictions(tgt, attn_feats, clip_bk_feats)
        masks = self.forward_mask_predictions(tgt, mask_feats)
        predictions_class.append(logits)
        predictions_mask.append(masks)

        frame_tgt = einops.rearrange(frame_embeds, 'b t q c -> t (b q) c')

        t_tgt = frame_tgt

        for i in range(self.num_layers):

            long_tgt = self.long_aggregate_layers[i](t_tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None)
            short_tgt = long_tgt.permute(1, 2, 0)  # (bq, c, t)
            short_tgt = (self.short_aggregate_layers[i](short_tgt) + short_tgt).transpose(1, 2)
            t_tgt = self.aggregate_norms[i](short_tgt)
            t_tgt = t_tgt.permute(1, 0, 2)
            t_tgt = einops.rearrange(t_tgt, 't (b q) c -> q (b t) c', b=bs)
            t_tgt = self.ffn_layers[i](t_tgt)

            sep_frame_mem = einops.rearrange(t_tgt, 't (b q) c -> (q t) b c', b=bs)
            sep_frame_mem = sep_frame_mem.repeat(1, t, 1)

            tgt = self.tgt_ca_layers[i](tgt, sep_frame_mem, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=query_pos)
            tgt = self.tgt_sa_layers[i](tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos)
            tgt = self.tgt_ffn_layers[i](tgt)

            logits = self.forward_class_predictions(tgt, attn_feats, clip_bk_feats)
            masks = self.forward_mask_predictions(tgt, mask_feats)
            predictions_class.append(logits)
            predictions_mask.append(masks)

            t_tgt = einops.rearrange(t_tgt, 'q (b t) c -> t (b q) c', b=bs)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', b=bs)

        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', b=bs)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
        }
        self.adapter = None
        self.text_feats = None
        return out

    def forward_mask_predictions(self, output, mask_feats):
        output = self.decode_norm(output).transpose(1, 0)

        mask_embed = self.mask_embed(output)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feats)

        return masks

    def forward_class_predictions(self, output, attn_feats, clip_bk_feats):
        output = self.decode_norm(output).transpose(1, 0)

        attn_embed = self.attn_embed(output)
        attn_biases  = torch.einsum("bqc,bnchw->bnqhw", attn_embed, attn_feats)

        clip_feats = self.adapter.post_encode_image(clip_bk_feats, attn_biases)
        logits = self.adapter.cal_sim_logits(self.text_feats, clip_feats)

        return logits

    @torch.jit.unused
    def _set_aux_loss(self, out_attn_biases, out_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(out_attn_biases[:-1], out_masks[:-1])]


class TemporalInstanceResampler(nn.Module):

    def __init__(self, hidden_dim=256, feed_dim=2048, nheads=8, nlayers=6):
        super().__init__()
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = nlayers

        self.long_aggregate_layers = nn.ModuleList()
        self.short_aggregate_layers = nn.ModuleList()
        self.aggregate_norms = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.long_aggregate_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.aggregate_norms.append(nn.LayerNorm(hidden_dim))

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feed_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decode_norm = nn.LayerNorm(hidden_dim)

        # output FFNs
        self.attn_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.adapter = None
        self.text_feats = None

    def forward(self, frame_embeds, mask_feats, attn_feats, adapter=None, clip_bk_feats=None, text_feats=None):
        self.adapter = adapter
        self.text_feats = text_feats

        predictions_class = []
        predictions_mask = []

        bs, t, q = frame_embeds.shape[:3]
        outputs_class, outputs_mask = self.forward_prediction_heads(einops.rearrange(frame_embeds, 'b t q c -> q (b t) c'), mask_feats, attn_feats, clip_bk_feats)

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        frame_tgt = einops.rearrange(frame_embeds, 'b t q c -> t (b q) c')

        temporal_tgt = frame_tgt

        for i in range(self.num_layers):
            long_tgt = self.long_aggregate_layers[i](
                temporal_tgt, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            short_tgt = long_tgt.permute(1, 2, 0)  # (bq, c, t)
            short_tgt = (self.short_aggregate_layers[i](short_tgt) + short_tgt).transpose(1, 2)

            temporal_tgt = self.aggregate_norms[i](short_tgt)
            temporal_tgt = temporal_tgt.permute(1, 0, 2)

            temporal_tgt = self.transformer_ffn_layers[i](temporal_tgt)

            temporal_tgt = einops.rearrange(temporal_tgt, 't (b q) c -> q (b t) c', b=bs)

            outputs_class, outputs_mask = self.forward_prediction_heads(temporal_tgt, mask_feats, attn_feats, clip_bk_feats)

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            temporal_tgt = einops.rearrange(temporal_tgt, 'q (b t) c -> t (b q) c', b=bs)

        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', b=bs)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', b=bs)

        pred_embeds = self.decode_norm(temporal_tgt)
        pred_embeds = einops.rearrange(pred_embeds, 't (b q) c -> b t q c', b=bs)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_embeds': pred_embeds,
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
        }
        self.adapter = None
        self.text_feats = None
        return out

    def forward_prediction_heads(self, output, mask_feats, attn_feats, clip_bk_feats):
        output = self.decode_norm(output).transpose(1, 0)

        mask_embed = self.mask_embed(output)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feats)

        attn_embed = self.attn_embed(output)
        attn_biases  = torch.einsum("bqc,bnchw->bnqhw", attn_embed, attn_feats)

        clip_feats = self.adapter.post_encode_image(clip_bk_feats, attn_biases)
        logits = self.adapter.cal_sim_logits(self.text_feats, clip_feats)

        return logits, masks

    @torch.jit.unused
    def _set_aux_loss(self, out_attn_biases, out_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(out_attn_biases[:-1], out_masks[:-1])]


class RawTemporalInstanceResampler(nn.Module):

    def __init__(self, hidden_dim=256, feed_dim=2048, nheads=8, nlayers=6, window_size=10):
        super().__init__()
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = nlayers
        self.window_size = window_size

        self.long_aggregate_layers = nn.ModuleList()
        self.short_aggregate_layers = nn.ModuleList()
        self.aggregate_norms = nn.ModuleList()

        self.pre_transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.post_transformer_cross_attention_layers = nn.ModuleList()
        self.cross_frame_norms = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.long_aggregate_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.aggregate_norms.append(nn.LayerNorm(hidden_dim))

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feed_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decode_norm = nn.LayerNorm(hidden_dim)

        # output FFNs
        self.attn_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.adapter = None
        self.text_feats = None

    def forward(self, src, pos, size_list, frame_embeds, mask_feats, attn_feats, adapter=None, clip_bk_feats=None, text_feats=None):
        self.adapter = adapter
        self.text_feats = text_feats

        predictions_class = []
        predictions_mask = []

        window_flag = False
        if isinstance(frame_embeds, list):
            window_flag = True
            assert len(src) == len(pos) == len(mask_feats) == len(attn_feats) == len(clip_bk_feats)
            bs, t, q = frame_embeds[0].shape[0], sum([f.shape[1] for f in frame_embeds]), frame_embeds[0].shape[2]
            num_feature_level = len(size_list)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads_list([einops.rearrange(f, 'b t q c -> q (b t) c') for f in frame_embeds], mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size=size_list[0])
            frame_embeds = torch.cat(frame_embeds, dim=1)
        else:
            bs, t, q = frame_embeds.shape[:3]
            num_feature_level = len(size_list)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(einops.rearrange(frame_embeds, 'b t q c -> q (b t) c'), mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size=size_list[0])

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        frame_tgt = einops.rearrange(frame_embeds, 'b t q c -> t (b q) c')

        temporal_tgt = frame_tgt

        for i in range(self.num_layers):
            long_tgt = self.long_aggregate_layers[i](
                temporal_tgt, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            short_tgt = long_tgt.permute(1, 2, 0)  # (bq, c, t)
            short_tgt = (self.short_aggregate_layers[i](short_tgt) + short_tgt).transpose(1, 2)

            temporal_tgt = self.aggregate_norms[i](short_tgt)
            temporal_tgt = temporal_tgt.permute(1, 0, 2)

            temporal_tgt = einops.rearrange(temporal_tgt, 't (b q) c -> q (b t) c', b=bs)

            level_index = i % num_feature_level
            next_level_index = (i + 1) % num_feature_level

            if not self.training and window_flag:
                count = 0
                tgt_list = []
                for s, p in zip(src, pos):
                    start_idx = count * self.window_size
                    end_idx = (count + 1) * self.window_size
                
                    iter_tgt = temporal_tgt[:, start_idx:end_idx]
                    iter_attn_mask = attn_mask[start_idx*self.num_heads:end_idx*self.num_heads]

                    tgts = self.resample_infer(iter_tgt, i, iter_attn_mask, s[level_index], p[level_index])
                    count += 1

                    tgt_list.append(tgts)

                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads_list(tgt_list, mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size=size_list[next_level_index])
                temporal_tgt = torch.cat(tgt_list, dim=1)
            else:
                temporal_tgt = self.resample_infer(temporal_tgt, i, attn_mask, src[level_index], pos[level_index])
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(temporal_tgt, mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size=size_list[next_level_index])

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            temporal_tgt = einops.rearrange(temporal_tgt, 'q (b t) c -> t (b q) c', b=bs)

        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', b=bs)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', b=bs)

        pred_embeds = self.decode_norm(temporal_tgt)
        pred_embeds = einops.rearrange(pred_embeds, 't (b q) c -> b t q c', b=bs)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_embeds': pred_embeds,
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
        }
        self.adapter = None
        self.text_feats = None
        return out

    def resample_infer(self, tgt, mi, attn_mask, src, pos):
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        tgt = self.transformer_cross_attention_layers[mi](
            tgt, src,
            memory_mask=None,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=pos,
            query_pos=None
        )

        tgt = self.transformer_self_attention_layers[mi](
            tgt, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=None
        )

        # FFN
        tgt = self.transformer_ffn_layers[mi](tgt)

        return tgt

    def window_resample_infer(self, tgt, mi, attn_mask, src, pos, mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size):
        bt = tgt.shape[1]
        iters = (bt // self.window_size) - 1
        if bt % self.window_size != 0:
            iters += 1

        tgt_list = []
        class_list = []
        mask_list = []
        attn_mask_list = []

        for i in range(iters):
            start_idx = i * self.window_size
            end_idx = (i+1) * self.window_size

            iter_tgt = tgt[:, start_idx:end_idx]

            iter_attn_mask = attn_mask[start_idx*self.num_heads:end_idx*self.num_heads]
            iter_src = src[:, start_idx:end_idx]
            iter_pos = pos[:, start_idx:end_idx]
            imask_feats = mask_feats[start_idx:end_idx]
            iattn_feats = attn_feats[start_idx:end_idx]

            iclip_bk_feats = {k: v[:, start_idx:end_idx] if 'cls_token' in str(k) else v[start_idx:end_idx] for k, v in clip_bk_feats.items()}

            # print(i, start_idx, end_idx, iter_tgt.shape, iter_src.shape, iter_pos.shape, imask_feats.shape)

            itgt, ioutputs_class, ioutputs_mask, iattn_mask = self.resample_infer(iter_tgt, mi, iter_attn_mask, iter_src, iter_pos, imask_feats, iattn_feats, iclip_bk_feats, attn_mask_target_size)

            tgt_list.append(itgt)
            class_list.append(ioutputs_class)
            mask_list.append(ioutputs_mask)
            attn_mask_list.append(iattn_mask)

        return torch.cat(tgt_list, dim=1), torch.cat(class_list, dim=0), torch.cat(mask_list, dim=0), torch.cat(attn_mask_list, dim=0)

    def frame_move(self, feats, bs):
        # hw, bt, c
        feats = einops.rearrange(feats, 'n (b t) c -> b t n c', b=bs)

        slow_feats = torch.roll(feats, 1, dims=1)
        fast_feats = torch.roll(feats, -1, dims=1)

        slow_feats[:, :1, :, :] = 0
        fast_feats[:, -1:, :, :] = 0

        feats = einops.rearrange(feats, 'b t n c -> n (b t) c')
        slow_feats = einops.rearrange(slow_feats, 'b t n c -> n (b t) c')
        fast_feats = einops.rearrange(fast_feats, 'b t n c -> n (b t) c')

        return slow_feats, feats, fast_feats

    def forward_prediction_heads(self, output, mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size):
        output = self.decode_norm(output).transpose(1, 0)

        mask_embed = self.mask_embed(output)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feats)

        attn_embed = self.attn_embed(output)
        attn_biases  = torch.einsum("bqc,bnchw->bnqhw", attn_embed, attn_feats)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(masks, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        clip_feats = self.adapter.post_encode_image(clip_bk_feats, attn_biases)
        logits = self.adapter.cal_sim_logits(self.text_feats, clip_feats)

        return logits, masks, attn_mask

    def forward_prediction_heads_list(self, output, mask_feats, attn_feats, clip_bk_feats, attn_mask_target_size):
        logits_list = []
        masks_list = []
        attn_masks_list = []
        for o, m, a, c in zip(output, mask_feats, attn_feats, clip_bk_feats):
            o = self.decode_norm(o).transpose(1, 0)

            mask_embed = self.mask_embed(o)
            masks = torch.einsum("bqc,bchw->bqhw", mask_embed, m)

            attn_embed = self.attn_embed(o)
            attn_biases  = torch.einsum("bqc,bnchw->bnqhw", attn_embed, a)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(masks, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            logits = self.adapter.get_sim_logits_frame(c, attn_biases, self.text_feats)

            logits_list.append(logits)
            masks_list.append(masks)
            attn_masks_list.append(attn_mask)

        return torch.cat(logits_list, dim=0), torch.cat(masks_list, dim=0), torch.cat(attn_masks_list, dim=0)

    @torch.jit.unused
    def _set_aux_loss(self, out_attn_biases, out_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(out_attn_biases[:-1], out_masks[:-1])]
