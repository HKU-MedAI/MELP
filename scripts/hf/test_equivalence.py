"""Check the exported HF folder reproduces the Lightning MELPModel it came from.

    HF_HUB_OFFLINE=1 python test_equivalence.py <ckpt> <hf_dir>

Compares, on CPU in float32: tokenisation, the 131 CKEPE class embeddings built the
way scripts/zeroshot/test_zeroshot.py builds them, and ECG embeddings for real
PTB-XL records plus random input.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from melp.models.melp_model import MELPModel as LightningMELP
from melp.datasets.finetune_datamodule import ECGDataModule
from melp.paths import RAW_DATA_PATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zeroshot"))
from test_zeroshot import get_class_emd  # noqa: E402

torch.manual_seed(0)


def max_diff(a, b):
    return (a - b).abs().max().item()


def main(ckpt, hf_dir):
    ref = LightningMELP.load_from_checkpoint(
        ckpt, map_location="cpu", weights_only=False,
        ecg_encoder_weight="").eval()
    hf = AutoModel.from_pretrained(hf_dir, trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    prompts = json.load(open(f"{hf_dir}/prompts/CKEPE_prompt.json"))
    texts = list(prompts.values())

    # 1. tokenisation
    r = ref._tokenize([t.lower() for t in texts])
    ids, mask = hf.tokenize(tok, [t.lower() for t in texts])
    assert torch.equal(r.input_ids, ids) and torch.equal(r.attention_mask, mask), "tokenisation differs"
    print(f"tokenisation identical for {len(texts)} prompts, shape {tuple(ids.shape)}")

    # 2. zero-shot class weights
    with torch.no_grad():
        w_ref = get_class_emd(ref, texts, device="cpu")
        w_hf = hf.zeroshot_weights(tok, texts)
    d_text = max_diff(w_ref, w_hf)
    print(f"class weights {tuple(w_hf.shape)}: max |diff| = {d_text:.2e}")

    # 3. ECG embeddings: real PTB-XL test records and random input
    loader = ECGDataModule(dataset_dir=str(RAW_DATA_PATH), dataset_name="ptbxl_rhythm",
                           batch_size=16, num_workers=2, train_data_pct=1).test_dataloader()
    ecg = torch.cat([next(iter(loader))["ecg"], torch.rand(4, 12, 5000)])
    with torch.no_grad():
        e_ref = ref.encode_ecg(ecg, normalize=True, proj_contrast=True)
        e_hf = hf.encode_ecg(ecg)
        out = hf(ecg=ecg)
        _, beat_ref, tok_ref = ref._encode_ecg(ecg)
    d_ecg = max_diff(e_ref, e_hf)
    d_beat = max_diff(beat_ref, out["ecg_beat_emb"])
    d_tok = max_diff(tok_ref, out["ecg_token_emb"])
    print(f"ECG emb {tuple(e_hf.shape)}: max |diff| = {d_ecg:.2e}   "
          f"beat {d_beat:.2e}   token {d_tok:.2e}")

    # 4. scores the two produce for the same pairs
    s_ref = (e_ref @ w_ref).numpy()
    s_hf = (e_hf @ w_hf).numpy()
    print(f"score matrix {s_hf.shape}: max |diff| = {np.abs(s_ref - s_hf).max():.2e}, "
          f"argmax agree {np.mean(s_ref.argmax(1) == s_hf.argmax(1)):.0%}")

    tol = 1e-5
    ok = max(d_text, d_ecg, d_beat, d_tok) < tol
    print("PASS" if ok else f"FAIL (tolerance {tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
