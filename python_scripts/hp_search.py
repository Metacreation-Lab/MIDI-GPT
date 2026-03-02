import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))
import midigpt

import json
import time
import datetime
import logging

import numpy as np
import torch
import torch.distributed as dist
import optuna

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from subprocess import check_output

from custom_models import GPT2Encoder, GPT2LMHeadModelCont, GPT2LMHeadModelContConfig
from train_dataset import CustomDataset


logger = logging.getLogger(__name__)


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def build_model(args, vocab_size):
    if args.arch == "gpt2":
        config = GPT2Config().from_json_file(args.config)
        model_cls = GPT2LMHeadModel
    elif args.arch == "xl":
        config = TransfoXLConfig().from_json_file(args.config)
        model_cls = TransfoXLLMHeadModel
    elif args.arch == "metric":
        config = GPT2Config().from_json_file(args.config)
        model_cls = GPT2Encoder
    elif args.arch == "control":
        config = GPT2LMHeadModelContConfig().from_json_file(args.config)
        encoder = midigpt.getEncoder(midigpt.getEncoderType(args.encoding))
        config.n_control_dim = encoder.config.embed_dim
        model_cls = GPT2LMHeadModelCont
    elif args.arch == "bert":
        config = BertConfig().from_json_file(args.config)
        model_cls = BertForMaskedLM
    else:
        raise NotImplementedError(f"Unknown architecture: {args.arch}")
    config.vocab_size = vocab_size
    return model_cls(config)


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, args, val_dataset, pruning_interval):
        self.trial = trial
        self.cli_args = args
        self.val_dataset = val_dataset
        self.pruning_interval = pruning_interval

    def on_step_end(self, train_args, state, control, model=None, **kwargs):
        if state.global_step % self.pruning_interval != 0 or state.global_step == 0:
            return
        val_loss = compute_validation_loss(model, self.val_dataset, self.cli_args)
        if is_main_process():
            self.trial.report(val_loss, state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial {self.trial.number} pruned at step {state.global_step}"
                )


def compute_validation_loss(model, val_dataset, args):
    model.eval()
    device = next(model.parameters()).device
    unwrapped = model.module if hasattr(model, "module") else model

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataset:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = unwrapped(**inputs)
            loss = outputs[0].mean()
            total_loss += loss.item()
            n_batches += 1

    model.train()

    avg_loss = total_loss / max(n_batches, 1)

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss, n_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        avg_loss = loss_tensor[0].item() / world_size

    return avg_loss


def make_objective(args, vocab_size, encoder_mode):

    valid_eff_batch_sizes = []
    micro_batch = args.per_device_batch_size * args.ngpu
    candidate = args.eff_batch_min
    while candidate <= args.eff_batch_max:
        if candidate % micro_batch == 0 and candidate // micro_batch >= 1:
            valid_eff_batch_sizes.append(candidate)
        candidate *= 2
    if not valid_eff_batch_sizes:
        raise ValueError(
            f"No valid effective batch sizes in [{args.eff_batch_min}, {args.eff_batch_max}] "
            f"for per_device_batch_size={args.per_device_batch_size} x ngpu={args.ngpu}. "
            f"Micro-batch={micro_batch}, need effective_batch_size divisible by it."
        )

    def objective(trial):
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        effective_batch_size = trial.suggest_categorical(
            "effective_batch_size", valid_eff_batch_sizes
        )
        num_epochs = trial.suggest_int("num_epochs", args.epochs_min, args.epochs_max)

        accum_steps = effective_batch_size // (args.per_device_batch_size * args.ngpu)

        trial_seed = args.seed + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)

        if is_main_process():
            logger.info(
                "Trial %d: lr=%.2e, eff_batch=%d, epochs=%d, accum=%d, seed=%d",
                trial.number, lr, effective_batch_size, num_epochs, accum_steps, trial_seed,
            )

        model = build_model(args, vocab_size)

        train_dataset = CustomDataset(
            split_id=0,
            is_training=True,
            batch_size=args.per_device_batch_size * accum_steps,
            dataset=args.dataset,
            num_bars=args.num_bars,
            min_tracks=args.min_tracks,
            max_tracks=args.max_tracks,
            max_seq_len=args.max_seq_len,
            expressive=args.expressive,
            no_max_length=args.no_max_length,
            resolution=args.resolution,
            encoding=args.encoding,
            pad_value=args.pad_value,
            arch=args.arch,
            accum_steps=accum_steps,
            batches_per_epoch=args.batches_per_epoch,
        )

        val_dataset = CustomDataset(
            split_id=2,
            is_training=False,
            batch_size=args.per_device_batch_size,
            dataset=args.dataset,
            num_bars=args.num_bars,
            min_tracks=args.min_tracks,
            max_tracks=args.max_tracks,
            max_seq_len=args.max_seq_len,
            expressive=args.expressive,
            no_max_length=args.no_max_length,
            resolution=args.resolution,
            encoding=args.encoding,
            pad_value=args.pad_value,
            arch=args.arch,
            accum_steps=1,
            batches_per_epoch=args.val_batches,
        )

        output_dir = os.path.join(args.output, f"hp_trial_{trial.number}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            lr_scheduler_type="linear",
            warmup_steps=0,
            gradient_accumulation_steps=accum_steps,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=args.log_steps,
            report_to="none",
            disable_tqdm=True,
            prediction_loss_only=True,
            skip_memory_metrics=True,
            seed=trial_seed,
        )

        pruning_callback = OptunaPruningCallback(
            trial=trial,
            args=args,
            val_dataset=val_dataset,
            pruning_interval=args.pruning_interval,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[pruning_callback],
        )
        trainer.get_train_dataloader = lambda: train_dataset
        trainer.get_eval_dataloader = lambda *_a, **_kw: val_dataset

        trainer.train()

        final_val_dataset = CustomDataset(
            split_id=2,
            is_training=False,
            batch_size=args.per_device_batch_size,
            dataset=args.dataset,
            num_bars=args.num_bars,
            min_tracks=args.min_tracks,
            max_tracks=args.max_tracks,
            max_seq_len=args.max_seq_len,
            expressive=args.expressive,
            no_max_length=args.no_max_length,
            resolution=args.resolution,
            encoding=args.encoding,
            pad_value=args.pad_value,
            arch=args.arch,
            accum_steps=1,
            batches_per_epoch=args.val_batches,
        )
        val_loss = compute_validation_loss(
            trainer.model, final_val_dataset, args
        )

        if is_main_process():
            log_entry = {
                "trial": trial.number,
                "lr": lr,
                "effective_batch_size": effective_batch_size,
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": accum_steps,
                "val_loss": val_loss,
                "seed": trial_seed,
                "status": "complete",
            }
            log_path = os.path.join(args.output, "hp_search_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info("Trial %d finished: val_loss=%.4f", trial.number, val_loss)

        barrier()
        return val_loss

    return objective


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for MIDI-GPT")

    # Model / data args (same as train.py)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--encoding", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pad_value", type=int, default=-100)
    parser.add_argument("--expressive", action="store_true")
    parser.add_argument("--num_bars", type=int, default=4)
    parser.add_argument("--min_tracks", type=int, default=1)
    parser.add_argument("--max_tracks", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--no_max_length", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--delta_resolution", type=int, default=1920)
    parser.add_argument("--abs_pos_vocab_size", type=int, default=196)
    parser.add_argument("--delta_vocab_size", type=int, default=96)

    # GPU / training infra
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--batches_per_epoch", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=100)

    # Optuna args
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="midigpt_hp_search")
    parser.add_argument("--storage", type=str, default="sqlite:///hp_search.db")
    parser.add_argument("--pruning_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # Search ranges
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--eff_batch_min", type=int, default=16)
    parser.add_argument("--eff_batch_max", type=int, default=256)
    parser.add_argument("--epochs_min", type=int, default=10)
    parser.add_argument("--epochs_max", type=int, default=100)

    # Validation
    parser.add_argument("--val_batches", type=int, default=50)

    # Output
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--label", type=str, default="hp_search")

    args = parser.parse_args()
    args.expressive = (args.encoding == "EXPRESSIVE_ENCODER") and args.expressive

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Encoder setup
    encoder_mode = midigpt.getEncoderType(args.encoding)
    assert encoder_mode is not midigpt.ENCODER_TYPE.NO_ENCODER
    encoder = midigpt.getEncoder(encoder_mode)
    if args.expressive:
        encoder.set_scheme(
            args.resolution, args.delta_resolution,
            args.delta_vocab_size, args.abs_pos_vocab_size,
        )
    vocab_size = encoder.vocab_size()

    # Output directory
    if not args.output:
        args.output = os.path.join(
            os.environ.get("SCRATCH", "."),
            ".midigpt", "hp_search", args.label,
        )
    os.makedirs(args.output, exist_ok=True)

    # Copy dataset to SLURM_TMPDIR if available
    if os.getenv("SLURM_TMPDIR") is not None:
        from shutil import copyfile
        tmpdir = os.getenv("SLURM_TMPDIR")
        dataset_path = os.path.join(tmpdir, os.path.basename(args.dataset))
        if not os.path.exists(dataset_path):
            copyfile(args.dataset, dataset_path)
            copyfile(args.dataset + ".header", dataset_path + ".header")
        args.dataset = dataset_path

    if is_main_process():
        logger.info("Study: %s", args.study_name)
        logger.info("Storage: %s", args.storage)
        logger.info("Vocab size: %d", vocab_size)
        logger.info("Output: %s", args.output)
        logger.info("Args: %s", json.dumps(vars(args), indent=2))

    # Create Optuna study (rank 0 creates, others load)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=args.pruning_interval * 2,
    )

    if is_main_process():
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            pruner=pruner,
            load_if_exists=True,
        )
    else:
        study = None

    barrier()

    if not is_main_process():
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )

    objective = make_objective(args, vocab_size, encoder_mode)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        catch=(optuna.TrialPruned,),
    )

    # Report results
    if is_main_process():
        best = study.best_trial
        result = {
            "best_trial": best.number,
            "best_val_loss": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "study_name": args.study_name,
        }
        logger.info("Best trial: %d", best.number)
        logger.info("Best val_loss: %.4f", best.value)
        logger.info("Best params: %s", json.dumps(best.params, indent=2))

        result_path = os.path.join(args.output, "best_hparams.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved best hyperparameters to %s", result_path)


if __name__ == "__main__":
    main()
