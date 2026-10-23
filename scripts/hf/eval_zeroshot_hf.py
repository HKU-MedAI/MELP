"""Six-dataset zero-shot AUROC using only an exported HF model.

    python eval_zeroshot_hf.py <hf_dir_or_repo_id>      # e.g. xjc1022/MELP-repro

Data loading reuses melp's ECGDataModule test splits (the paper's protocol); the
model, tokenizer and prompts all come from the HF folder or repo.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

from melp.datasets.finetune_datamodule import ECGDataModule
from melp.paths import RAW_DATA_PATH

SETS = ["ptbxl_rhythm", "ptbxl_form", "ptbxl_sub_class", "ptbxl_super_class", "icbeb", "chapman"]
PAPER = {"ptbxl_rhythm": 85.4, "ptbxl_form": 69.1, "ptbxl_sub_class": 81.2,
         "ptbxl_super_class": 76.2, "icbeb": 84.2, "chapman": 77.6}  # Table 3


def main(repo):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    prompt_file = (os.path.join(repo, "prompts/CKEPE_prompt.json") if os.path.isdir(repo)
                   else hf_hub_download(repo, "prompts/CKEPE_prompt.json"))
    prompts = json.load(open(prompt_file))

    scores = {}
    for name in SETS:
        t0 = time.time()
        loader = ECGDataModule(dataset_dir=str(RAW_DATA_PATH), dataset_name=name,
                               batch_size=64, num_workers=8, train_data_pct=1).test_dataloader()
        classes = loader.dataset.labels_name
        weights = model.zeroshot_weights(tokenizer, [prompts[c] for c in classes])
        preds = []
        with torch.no_grad():
            for batch in loader:
                preds.append((model.encode_ecg(batch["ecg"].to(device)) @ weights).cpu().numpy())
        pred, gt = np.concatenate(preds), loader.dataset.labels
        auroc = np.mean([roc_auc_score(gt[:, i], pred[:, i]) for i in range(len(classes))]) * 100
        scores[name] = auroc
        print(f"{name:18s} n={len(gt):5d} classes={len(classes):2d}  AUROC {auroc:6.2f}  "
              f"(paper {PAPER[name]:.1f})  [{time.time() - t0:.0f}s]", flush=True)

    print(f"{'AVERAGE':18s} AUROC {np.mean(list(scores.values())):6.2f}  "
          f"(paper {np.mean(list(PAPER.values())):.1f})")


if __name__ == "__main__":
    main(sys.argv[1])
