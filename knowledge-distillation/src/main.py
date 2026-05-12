import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_modules.datamodule import MyDataModule
from models.distillation import DistillationModel

def train_func(config):
    print("Initializing DataModule...")
    datamodule = MyDataModule(batch_size=config["batch_size"])

    print("Initializing Distillation Model...")
    model = DistillationModel(config)
    # Initialize with weights from the checkpoint, but use the new config
    # model = DistillationModel.load_from_checkpoint("checkpoints/last.ckpt", config=config)
    # model.compile(mode='default')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mdm-distill-{epoch:02d}-{validation_loss:.4f}",
        save_top_k=2,             # Keep the top 3 best models
        monitor="validation_loss", # Must match the exact key logged in validation_step
        mode="min",
        save_last=True            # Always save the latest epoch to resume easily
    )

    # Logs the learning rate to TensorBoard
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop = EarlyStopping(
        monitor="validation_loss",
        patience=10,
        mode="min"
    )

    # 4. Setup Logger
    # logger = TensorBoardLogger(save_dir="logs/", name="hypertesting", version="lr=2.3e-4")
    logger = TensorBoardLogger(save_dir="logs/", name="small_train", version="only_at_neck_concat_mae_correct")

    print("Setting up PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=1000,                  
        accelerator="auto", 
        devices=1,
        precision="bf16-mixed",         
        accumulate_grad_batches=max(1, 1024 // config["batch_size"]),      
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        logger=logger,
        log_every_n_steps=10,
        # profiler="simple",
        # fast_dev_run=True
    )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    # print("done")

def main():
    torch.set_float32_matmul_precision('medium')

    # # Increase TorchDynamo compile cache limits to avoid recompilation errors
    # torch._dynamo.config.cache_size_limit = 1024
    # if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
    #     torch._dynamo.config.accumulated_cache_size_limit = 1024

    pl.seed_everything(42, workers=True)

    config = {
        'learning_rate_backbone': 2.6e-5,
        'learning_rate': 2.6e-4,
        'betas_lr': (0.9, 0.999),
        'weight_decay': 0.1,
        'batch_size': 8,
        'alpha': 0.85,
        'beta': 0.15,
        'gamma': 0.1
    }

    train_func(config)

if __name__ == "__main__":
    main()