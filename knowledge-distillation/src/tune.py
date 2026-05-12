import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_modules.datamodule import MyDataModule
from models.distillation import DistillationModel

def tune_asha(train_fn_with_resources, num_epochs, search_space, num_samples):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        train_fn_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.abspath("ray_results"),
            name="mdm_distillation_tune",
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="validation_loss",
                checkpoint_score_order="min",
            ),
        ),
    )
    return tuner.fit()

def train_func(config):
    print("Initializing DataModule...")
    datamodule = MyDataModule()

    print("Initializing Distillation Model...")
    model = DistillationModel(config)
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

    tune_ckpt_callback = TuneReportCheckpointCallback()

    # 4. Setup Logger
    logger = TensorBoardLogger(save_dir="logs/", name="hypertesting", version="lr=1e-4_ori")

    print("Setting up PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=100,                  
        accelerator="auto", 
        devices=1,
        precision="bf16-mixed",         
        accumulate_grad_batches=1,      
        callbacks=[checkpoint_callback, lr_monitor, early_stop, tune_ckpt_callback],
        logger=logger,
        log_every_n_steps=100,
        # fast_dev_run=True
    )

    # Run learning rate finder
    tuner = pl.tuner.Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # trainer.fit(model, datamodule=datamodule)

    # trainer.test(model, datamodule=datamodule, ckpt_path="best")
    # print("done")

def main():
    torch.set_float32_matmul_precision('medium')

    pl.seed_everything(42, workers=True)

    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'batch_size': 16,
        'alpha': 0.85,
        'beta': 0.15,
        'gamma': 0.1
    }

    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16]),
        "weight_decay": tune.choice([0.01, 0.05, 0.1, 0.2]),
        'alpha': tune.uniform(0.5, 0.9),
        'beta': tune.uniform(0.1, 0.5),
        'gamma': tune.uniform(0.05, 0.5)
    }

    # # The maximum training epochs
    # num_epochs = 10

    # # Number of samples from parameter space
    # num_samples = 20

    # train_fn_with_resources = tune.with_resources(train_func, resources={"CPU": 24, "GPU": 1})

    # results = tune_asha(train_fn_with_resources, num_epochs, search_space, num_samples)

    # results.get_best_result(metric="validation_loss", mode="min")

    train_func(config)

if __name__ == "__main__":
    main()