import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_modules.datamodule import MyDataModule
from models.distillation import DistillationModel

def main():
    torch.set_float32_matmul_precision('medium')

    pl.seed_everything(42, workers=True)

    print("Initializing DataModule...")
    datamodule = MyDataModule()

    print("Initializing Distillation Model...")
    model = DistillationModel(learning_rate=1e-4)
    # model.compile(mode='default')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mdm-distill-{epoch:02d}-{validation_loss:.4f}",
        save_top_k=3,             # Keep the top 3 best models
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
    logger = TensorBoardLogger(save_dir="logs/", name="mdm_distillation")

    print("Setting up PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=50,                  
        accelerator="auto", 
        devices=1,
        precision="bf16-mixed",         
        accumulate_grad_batches=2,      
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        logger=logger,
        log_every_n_steps=10            # Update TensorBoard every 10 batches
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path="last")

    trainer.test(model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()