"""Convert a MELP Lightning checkpoint into a Hugging Face repo folder for zero-shot use.

    HF_HUB_OFFLINE=1 python export_melp.py <ckpt> <out_dir>

Keeps the ECG tower, the text tower (BERT without its MLM head), the text projection
and the logit scale; drops the captioning decoder and the sentence projection, which
only serve pretraining losses.
"""
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertConfig

from melp.paths import PROMPT_PATH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from melp_hf.configuration_melp import MELPConfig  # noqa: E402
from melp_hf.modeling_melp import MELPModel  # noqa: E402

PROMPT_DIR = PROMPT_PATH.parent
DROPPED_PREFIXES = ("lm_model.cls.", "text_decoder.", "sent_proj.")


def convert_state_dict(sd):
    out, dropped = {}, []
    for k, v in sd.items():
        if k.startswith("ecg_encoder.") or k.startswith("proj_t.") or k == "logit_scale":
            out[k] = v
        elif k.startswith("lm_model.bert."):
            out["text_encoder." + k[len("lm_model.bert."):]] = v
        elif k.startswith(DROPPED_PREFIXES):
            dropped.append(k)
        else:
            raise KeyError(f"unexpected checkpoint key {k}")
    return out, dropped


def main(ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    hp = ckpt["hyper_parameters"]
    assert hp["ecg_encoder_name"] == "ecgfm" and hp["text_encoder_mode"] == "causal", hp

    tokenizer = AutoTokenizer.from_pretrained(hp["text_encoder_name"])
    tokenizer.add_special_tokens({"bos_token": "[BOS]"})

    text_config = BertConfig.from_pretrained(hp["text_encoder_name"])
    text_config.vocab_size = len(tokenizer)
    text_config.is_decoder = True        # pretraining used AutoModelForCausalLM(is_decoder=True)
    text_config.architectures = None
    text_config._name_or_path = ""

    config = MELPConfig(
        model_size="small",
        shared_emb_dim=hp["shared_emb_dim"],
        n_queries_contrast=hp["n_queries_contrast"],
        num_leads=hp["num_leads"],
        text_config=text_config.to_dict(),
        max_seq_len=hp["max_seq_len"],
        text_cls_token_id=tokenizer.cls_token_id,
    )
    MELPConfig.register_for_auto_class()
    MELPModel.register_for_auto_class("AutoModel")
    model = MELPModel(config)

    sd, dropped = convert_state_dict(ckpt["state_dict"])
    sd = {k: v.clone() for k, v in sd.items()}  # detach from the mmap'd file
    model.load_state_dict(sd, strict=True)
    print(f"loaded {len(sd)} tensors, dropped {len(dropped)} "
          f"({sorted({k.split('.')[0] + '.' + k.split('.')[1] for k in dropped})})")
    n = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n / 1e6:.1f}M")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    (out / "prompts").mkdir(exist_ok=True)
    for f in ("CKEPE_prompt.json", "dataset_class_names.json"):
        shutil.copy(PROMPT_DIR / f, out / "prompts" / f)
    print(f"wrote {out}: {sorted(os.listdir(out))}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
