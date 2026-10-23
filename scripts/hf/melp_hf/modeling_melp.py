from collections import OrderedDict
from typing import Callable, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from fairseq_signals_backbone.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel, Wav2Vec2CMSCConfig
from transformers import BertModel, PreTrainedModel

from .configuration_melp import MELPConfig


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class _Wav2Vec2CMSCModel(Wav2Vec2CMSCModel):
    # transformers 4.5x loads each tensor with module.load_state_dict(..., assign=True),
    # which fairseq's override does not accept. These weights are already in the current
    # format, so fairseq's upgrade step has nothing to do; use torch's loader directly.
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


_ECG_SIZES = {
    "small": dict(encoder_embed_dim=768, encoder_attention_heads=12, encoder_layers=8, encoder_ffn_embed_dim=3072),
    "base": dict(encoder_embed_dim=768, encoder_attention_heads=12, encoder_layers=12, encoder_ffn_embed_dim=3072),
    "large": dict(encoder_embed_dim=1024, encoder_attention_heads=16, encoder_layers=24, encoder_ffn_embed_dim=4096),
}


class ECGFMModel(nn.Module):
    """ECG tower of MELP: wav2vec2-CMSC (CNN + Transformer) with attentional pooling heads.

    Module names match ``melp.models.ecgfm_model.ECGFMModel`` so a MELP checkpoint's
    ``ecg_encoder.*`` weights load into it unchanged.
    """

    def __init__(self,
                 model_size: str = "small",
                 shared_emb_dim: int = 256,
                 embed_dim_caption: int = 768,
                 use_attentional_pool_contrast: bool = True,
                 use_attentional_pool_caption: bool = True,
                 n_queries_contrast: int = 12,
                 n_queries_caption: int = 128,
                 attn_pooler_heads: int = 8,
                 norm_layer: nn.Module = nn.LayerNorm,
                 proj: str = "linear",
                 drop: float = 0.,
                 proj_bias: bool = False,
                 num_leads: int = 12):
        super().__init__()
        if model_size not in _ECG_SIZES:
            raise ValueError(f"Unknown model size: {model_size}")
        cfg = Wav2Vec2CMSCConfig(
            apply_mask=True,
            mask_prob=0.65,
            quantize_targets=True,
            final_dim=256,
            dropout_input=0.1,
            dropout_features=0.1,
            feature_grad_mult=0.1,
            in_d=num_leads,
            **_ECG_SIZES[model_size],
        )
        self.ecg_encoder = _Wav2Vec2CMSCModel(cfg)
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption

        head_layers = OrderedDict()
        prev_chs = cfg.encoder_embed_dim
        if use_attentional_pool_contrast:
            scale = prev_chs ** -0.5
            self.attn_pool_contrast = AttentionalPooler(
                d_model=shared_emb_dim,
                context_dim=prev_chs,
                n_head=attn_pooler_heads,
                n_queries=n_queries_contrast)
            self.ln_contrast = norm_layer(shared_emb_dim)
            self.proj_contrast = nn.Parameter(scale * torch.randn(shared_emb_dim, shared_emb_dim))
        else:
            if proj == 'linear':
                head_layers['drop'] = nn.Dropout(drop)
                head_layers['proj'] = nn.Linear(prev_chs, shared_emb_dim, bias=proj_bias)
            elif proj == 'mlp':
                head_layers['mlp'] = Mlp(prev_chs, 2 * shared_emb_dim, shared_emb_dim, drop=(drop, 0), bias=(True, proj_bias))
        self.head = nn.Sequential(head_layers)

        if use_attentional_pool_caption:
            self.attn_pool_caption = AttentionalPooler(
                d_model=embed_dim_caption, context_dim=prev_chs, n_head=attn_pooler_heads, n_queries=n_queries_caption)
            self.ln_caption = norm_layer(embed_dim_caption)

        # Unused at inference; kept so the checkpoint loads with strict key matching.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.bn = nn.BatchNorm1d(prev_chs, affine=False)

    def _encode_ecg(self, ecg):
        assert ecg.dim() == 3, "Input tensor must be 3D (batch, leads, samples)"
        ecg_out = self.ecg_encoder(source=ecg.float(), mask=False, features_only=True)
        features = ecg_out["x"].float()

        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(features)
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast.unsqueeze(0)
            pooled_beat = pooled.clone()
            pooled = torch.mean(pooled, dim=1)
        else:
            pooled = self.head(torch.mean(features, dim=1))
            pooled_beat = None

        tokens = None
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(features)
            tokens = self.ln_caption(tokens)

        return pooled, pooled_beat, tokens


def tokenize(tokenizer, texts: Sequence[str], max_seq_len: int = 128):
    """Tokenise text the way MELP did during pretraining.

    Prepends ``[BOS]``, pads to ``max_seq_len`` and drops the leading ``[CLS]``;
    :meth:`MELPModel.encode_text` then appends ``[CLS]`` after the padding and reads
    the text embedding from that last position. Padding to the full length is part
    of the recipe -- it fixes the position the embedding is read from.
    """
    texts = [f"[BOS] {t}" for t in texts]
    out = tokenizer(texts, add_special_tokens=True, truncation=True, max_length=max_seq_len,
                    padding="max_length", return_tensors="pt")
    return out["input_ids"][:, 1:], out["attention_mask"][:, 1:]


class MELPModel(PreTrainedModel):
    """MELP ECG and text towers, enough for zero-shot classification and retrieval.

    ``encode_ecg`` and ``encode_text`` map into the same 256-d space; cosine
    similarity between them is the zero-shot score. The captioning decoder used
    during pretraining is not included.
    """
    config_class = MELPConfig
    base_model_prefix = "melp"

    def __init__(self, config: MELPConfig):
        super().__init__(config)
        self.ecg_encoder = ECGFMModel(
            model_size=config.model_size,
            shared_emb_dim=config.shared_emb_dim,
            embed_dim_caption=config.embed_dim_caption,
            use_attentional_pool_contrast=config.use_attentional_pool_contrast,
            use_attentional_pool_caption=config.use_attentional_pool_caption,
            n_queries_contrast=config.n_queries_contrast,
            n_queries_caption=config.n_queries_caption,
            attn_pooler_heads=config.attn_pooler_heads,
            proj=config.proj,
            drop=config.drop,
            proj_bias=config.proj_bias,
            num_leads=config.num_leads,
        )
        # is_decoder=True in text_config: pretraining ran BERT with a causal mask.
        self.text_encoder = BertModel(config.text_config, add_pooling_layer=False)
        hidden = config.text_config.hidden_size
        self.proj_t = nn.Sequential(
            nn.Linear(hidden, config.text_proj_hidden),
            nn.GELU(),
            nn.Linear(config.text_proj_hidden, config.shared_emb_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.post_init()

    def _init_weights(self, module):
        # Every weight comes from the checkpoint; nothing to initialise.
        pass

    def encode_ecg(self, ecg: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """(B, 12, 5000) ECG scaled to [0, 1] -> (B, 256) rhythm-level embedding."""
        pooled, _, _ = self.ecg_encoder._encode_ecg(ecg)
        return F.normalize(pooled, dim=-1) if normalize else pooled

    def tokenize(self, tokenizer, texts: Sequence[str]):
        """(input_ids, attention_mask) for :meth:`encode_text`; see :func:`tokenize`."""
        return tokenize(tokenizer, texts, self.config.max_seq_len)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    normalize: bool = True) -> torch.Tensor:
        """Token ids from :func:`tokenize` -> (B, 256) text embedding."""
        cls = torch.full((input_ids.shape[0], 1), self.config.text_cls_token_id,
                         dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, cls), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(cls)), dim=1)
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = self.proj_t(hidden[:, -1])
        return F.normalize(emb, dim=-1) if normalize else emb

    @torch.no_grad()
    def zeroshot_weights(self, tokenizer, class_prompts: Sequence[Union[str, Sequence[str]]]) -> torch.Tensor:
        """One prompt (or a list of templates) per class -> (256, num_classes) classifier.

        Follows ``scripts/zeroshot/test_zeroshot.py``: lower-case each prompt, embed,
        normalise, average over templates, normalise again.
        """
        device = self.logit_scale.device
        weights = []
        for prompts in class_prompts:
            if isinstance(prompts, str):
                prompts = [prompts]
            ids, mask = self.tokenize(tokenizer, [p.lower() for p in prompts])
            emb = self.encode_text(ids.to(device), mask.to(device), normalize=True)
            emb = emb.mean(dim=0)
            weights.append(emb / emb.norm())
        return torch.stack(weights, dim=1)

    def forward(self, ecg: torch.Tensor = None, input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        out = {}
        if ecg is not None:
            pooled, beat, tokens = self.ecg_encoder._encode_ecg(ecg)
            out["ecg_emb"] = F.normalize(pooled, dim=-1)
            out["ecg_beat_emb"] = beat
            out["ecg_token_emb"] = tokens
        if input_ids is not None:
            out["text_emb"] = self.encode_text(input_ids, attention_mask)
        if ecg is not None and input_ids is not None:
            out["logits_per_ecg"] = self.logit_scale.exp() * out["ecg_emb"] @ out["text_emb"].t()
        return out
