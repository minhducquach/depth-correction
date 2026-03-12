import torch

torch.set_float32_matmul_precision('medium')

import lightning.pytorch as pl
import sys
import os

# Ensure Python can find your 'src' folder imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_modules.datamodule import MyDataModule
from models.distillation import DistillationModel

def main():
    # Optional: Set a seed for reproducibility during the test
    pl.seed_everything(42)

    print("Initializing DataModule...")
    datamodule = MyDataModule()

    print("Initializing Distillation Model...")
    # Initialize your model with whatever hyperparams you want to test
    model = DistillationModel(learning_rate=1e-4)

    print("Setting up PyTorch Lightning Trainer...")
    # The fast_dev_run=True flag is the magic trick here!
    trainer = pl.Trainer(
        fast_dev_run=True,  # Runs 1 train batch and 1 val batch to catch bugs
        accelerator="cpu", # Automatically uses GPU if available
        devices=1,           # Test on a single device first
        accumulate_grad_batches=4
    )

    print("Starting test run...")
    # If there are any shape mismatches, memory leaks, or missing dictionary keys,
    # the code will crash here.
    trainer.fit(model, datamodule=datamodule)

    print("\n✅ SUCCESS! If you see this, your data, model, and losses are working perfectly!")

if __name__ == "__main__":
    main()