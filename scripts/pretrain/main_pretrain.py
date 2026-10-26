import ipdb
from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from lightning import seed_everything, Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from melp.datasets.pretrain_datamodule import ECGTextDataModule
from melp.models.merl_model import MERLModel
from melp.models.ecgfm_model import ECGFMModel
from melp.models.melp_model import MELPModel
from melp.paths import ROOT_PATH as REPO_ROOT_DIR
from melp.paths import RAW_DATA_PATH

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain.py --num_devices 4 --train_data_pct 1 \
    --run_name melp_repro --model_name melp \
    --text_encoder_name fuyingw/heart_bert \
    --ecg_encoder_name ecgfm --ecg_encoder_weight "$MELP_ECGFM_PATH" \
    --lr 1e-4 --warmup_epochs 2 --batch_size 64 --max_epochs 100 \
    --early_stopping_patience 12 --val_check_interval 0.25 \
    --clip_loss_weight 1.0 --caption_loss_weight 2.0 --local_loss_weight 0.2
'''

def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    # Every DDP rank re-executes this script and computes its own timestamp, and the
    # format only has second resolution -- two runs started in the same second (or two
    # ranks of one run) silently share ckpt/log directories and write into each other's
    # metrics.csv. Pass --run_name to give a run one stable identity across its ranks.
    now = datetime.datetime.now(tz.tzlocal())
    extension = hparams.run_name or (
        f"melp_{hparams.model_name}_" + now.strftime("%Y_%m_%d_%H_%M_%S"))
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"logs/melp/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if hparams.model_name in ["merl", "melp"]:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val/mean_AUROC", dirpath=ckpt_dir,
                            save_last=True, mode="max", save_top_k=2,
                            auto_insert_metric_name=True),
            EarlyStopping(monitor="val/mean_AUROC", min_delta=0,
                        patience=hparams.early_stopping_patience, verbose=True, mode="max"),
        ]
    elif hparams.model_name in ["leadfusion", "vqnsp", "heartlang", "ecgfm"]:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val/loss", dirpath=ckpt_dir,
                            save_last=True, mode="min", save_top_k=2,
                            auto_insert_metric_name=True),
            EarlyStopping(monitor="val/loss", min_delta=0,
                        patience=hparams.early_stopping_patience, verbose=True, mode="min"),
        ]
    else:
        raise NotImplementedError
    logger_dir = os.path.join(REPO_ROOT_DIR, "logs/melp")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="melp", save_dir=logger_dir, name=extension)
    # CSV alongside wandb: per-epoch metrics land in a plain file that survives
    # without a wandb account and can be read while the run is still going.
    csv_logger = CSVLogger(save_dir=logger_dir, name="csv", version=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        # Every run so far peaked at the first validation point, i.e. the earliest
        # thing ever measured -- the real optimum may sit inside the first epoch.
        val_check_interval=hparams.val_check_interval,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision=32 if hparams.model_name == "ecgfm" else "bf16-mixed",
        callbacks=callbacks,
        logger=[wandb_logger, csv_logger]
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/{extension}/exp_logs")
    
    if hparams.model_name == "merl":
        datamodule = ECGTextDataModule(
            dataset_dir=str(RAW_DATA_PATH),
            dataset_list=["mimic-iv-ecg"],
            val_dataset_list=hparams.val_dataset_list,
            batch_size=hparams.batch_size,  
            num_workers=hparams.num_workers,
            train_data_pct=hparams.train_data_pct
        )
        model = MERLModel(**vars(hparams))
    elif hparams.model_name == "melp":
        datamodule = ECGTextDataModule(
            dataset_dir=str(RAW_DATA_PATH),
            dataset_list=["mimic-iv-ecg"],
            val_dataset_list=hparams.val_dataset_list,
            batch_size=hparams.batch_size,  
            num_workers=hparams.num_workers,
            train_data_pct=hparams.train_data_pct,
            use_rlm=True,
            ecg_source=hparams.ecg_source
        )
        model = MELPModel(**vars(hparams))
    elif hparams.model_name == "ecgfm":
        datamodule = ECGTextDataModule(
            dataset_dir=str(RAW_DATA_PATH),
            dataset_list=["mimic-iv-ecg"],
            val_dataset_list=None,
            batch_size=hparams.batch_size,  
            num_workers=hparams.num_workers,
            train_data_pct=hparams.train_data_pct,
            use_cmsc=True,
            use_rlm=True,
            ecg_source=hparams.ecg_source
        )
        model = ECGFMModel(**vars(hparams))
    else:
        raise NotImplementedError

    model.training_steps_per_epoch = len(datamodule.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices
    pprint(vars(hparams))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # tuner = Tuner(trainer)
    # Find optimal batch size
    # optimal_batch_size = tuner.scale_batch_size(model=model, datamodule=datamodule, init_val=128,
    #                                             mode="binsearch")
    # datamodule.batch_size = optimal_batch_size
    # print(f"Optimal batch size: {optimal_batch_size}")
    # Find optimal learning rate
    # lr_finder = tuner.lr_find(model=model, datamodule=datamodule, max_lr=1e-3)
    # model.lr = lr_finder.suggestion()
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretraining Multimodal ECG Foundation Model.")
    parser.add_argument("--model_name", type=str, default="merl",
                        choices=["merl", "ecgfm", "melp"])
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--run_name", type=str, default=None,
                        help="stable name for this run's ckpt/log dirs; required to "
                             "run two jobs concurrently (the timestamp fallback "
                             "collides at one-second resolution)")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="epochs without improvement before training stops")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="validate this often; <1.0 is a fraction of an epoch, "
                             "an int is a step count")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="lr warmup length; defaults to half of max_epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ecg_source", type=str, default="raw", choices=["raw", "processed"],
                        help="read pretraining waveforms from the raw wfdb records "
                             "(MELP_RAW_DATA_PATH) or the denoised .npy store "
                             "(MELP_PROCESSED_DATA_PATH/<dataset>/records)")
    parser.add_argument("--ecg_encoder_name", type=str, default="ecgfm")
    parser.add_argument("--ecg_encoder_weight", type=str, default="")
    parser.add_argument("--text_encoder_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--clip_loss_weight", type=float, default=1.)
    parser.add_argument("--caption_loss_weight", type=float, default=1.)
    parser.add_argument("--local_loss_weight", type=float, default=1.)
    parser.add_argument("--freeze_text_proj", action="store_true",
                        help="also freeze proj_t, pinning the zero-shot class weights")
    parser.add_argument("--text_encoder_mode", type=str, default="causal",
                        choices=["causal", "bidirectional"],
                        help="how the text encoder attends; MLM-trained encoders such "
                             "as heart_bert were trained bidirectionally")
    parser.add_argument("--num_freeze_layers", type=int, default=6,
                        help="text-encoder layers to freeze; at or above the encoder "
                             "depth the embedding table is frozen as well")
    parser.add_argument("--n_queries_contrast", type=int, default=12)
    parser.add_argument("--val_dataset_list", type=str, nargs="+", 
                        default=["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm", 
                                  "icbeb", "chapman"])

    hparams = parser.parse_args()

    # set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)
    main(hparams)