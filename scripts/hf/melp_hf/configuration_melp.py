from transformers import BertConfig, PretrainedConfig


class MELPConfig(PretrainedConfig):
    model_type = "melp"

    def __init__(self,
                 # ECG tower (wav2vec2-CMSC, same fields as MELPEncoderConfig)
                 model_size: str = "small",
                 shared_emb_dim: int = 256,
                 embed_dim_caption: int = 768,
                 use_attentional_pool_contrast: bool = True,
                 use_attentional_pool_caption: bool = True,
                 n_queries_contrast: int = 12,
                 n_queries_caption: int = 128,
                 attn_pooler_heads: int = 8,
                 proj: str = "linear",
                 drop: float = 0.,
                 proj_bias: bool = False,
                 num_leads: int = 12,
                 # text tower (BERT, run with a causal mask as in pretraining)
                 text_config: dict = None,
                 text_proj_hidden: int = 256,
                 max_seq_len: int = 128,
                 text_cls_token_id: int = 101,
                 **kwargs):
        self.model_size = model_size
        self.shared_emb_dim = shared_emb_dim
        self.embed_dim_caption = embed_dim_caption
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
        self.n_queries_contrast = n_queries_contrast
        self.n_queries_caption = n_queries_caption
        self.attn_pooler_heads = attn_pooler_heads
        self.proj = proj
        self.drop = drop
        self.proj_bias = proj_bias
        self.num_leads = num_leads

        if isinstance(text_config, BertConfig):
            text_config = text_config.to_dict()
        self.text_config = BertConfig(**(text_config or {}))
        self.text_proj_hidden = text_proj_hidden
        self.max_seq_len = max_seq_len
        # appended after padding; the text embedding is read from this position
        self.text_cls_token_id = text_cls_token_id
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output
