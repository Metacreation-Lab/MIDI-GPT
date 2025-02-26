import sys
import os
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
import midigpt

from transformers import *

import os
import json
import time
import torch

import datetime
import numpy as np
from tqdm import tqdm

from subprocess import check_output

from losses import sim_metric_loss, standard_loss
from custom_models import *
from train_dataset import *
from transformers import Trainer, TrainingArguments
from callbacks import MemoryUsageCallback, ProfilerCallback

if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--arch", type=str, required=True)
  parser.add_argument("--config", type=str, required=True)
  parser.add_argument("--encoding", type=str, required=True)
  parser.add_argument("--dataset", type=str, required=True)
  parser.add_argument("--pad_value", type=int, default=-100)

  parser.add_argument("--expressive", action="store_true")
  parser.add_argument("--num_bars", type=int, default=4)
  parser.add_argument("--min_tracks", type=int, default=2)
  parser.add_argument("--max_tracks", type=int, default=12)
  parser.add_argument("--max_seq_len", type=int, default=2048)
  parser.add_argument("--no_max_length", type=int, default=0)
  parser.add_argument("--resolution", type=int, default=12)
  parser.add_argument("--delta_resolution", type=int, default=1920)
  parser.add_argument("--abs_pos_vocab_size", type=int, default=196)
  parser.add_argument("--delta_vocab_size", type=int, default=96)

  parser.add_argument("--ngpu", type=int, default=4)
  parser.add_argument("--accum_steps", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--batches_per_epoch", type=int, default=1000)
  parser.add_argument("--lr", type=float, default=1e-4)

  parser.add_argument("--overwrite", type=int, default=1)
  parser.add_argument("--save_steps", type=int, default=5000)
  parser.add_argument("--log_steps", type=int, default=100)
  parser.add_argument("--step", type=int, default=0)
  parser.add_argument("--label", type=str, default="version3")
  parser.add_argument("--profiler_steps", type=int, default=50)

  parser.add_argument("--dry", action="store_true")
  parser.add_argument("--metric", action="store_true")

  parser.add_argument("--ckpt", type=str, default="")
  parser.add_argument("--ckpt_num", type=int, default=5000)
  parser.add_argument("--output", type=str, default="")
  parser.add_argument("--log", type=str, default="")

  parser.add_argument("--test_only", action="store_true")
  parser.add_argument("--memory_metrics", action="store_true")

  args = parser.parse_args()
  args.expressive = (args.encoding == "EXPRESSIVE_ENCODER") and args.expressive

  dataset_cls = CustomDataset
  loss_fn = standard_loss

  np.random.seed(int(time.time()))

  # determine vocab size
  date_str = datetime.datetime.now().strftime('%b_%d_%H_%M')
  encoder_mode = midigpt.getEncoderType(args.encoding)
  assert encoder_mode is not midigpt.ENCODER_TYPE.NO_ENCODER
  encoder = midigpt.getEncoder(encoder_mode)
  if args.expressive:
    encoder.set_scheme(args.resolution, args.delta_resolution, args.delta_vocab_size, args.abs_pos_vocab_size)
  vocab_size = encoder.vocab_size()

  current_git_commit_hash = check_output(["git", "rev-parse", "HEAD"], text=True).strip()

  load_checkpoint = False
  if args.ckpt == "":
    name = "_".join([args.encoding, args.arch, args.label, date_str, "num_bars", str(args.num_bars), str(args.max_tracks), "GIT_HASH", current_git_commit_hash])
  else:
    name = args.ckpt
    load_checkpoint = True

  if args.dry:
    while True:
      dataset = dataset_cls(split_id=0, is_training=True, **vars(args))
      for batch in tqdm(dataset,smoothing=0):
        np_inputs = batch["input_ids"].detach().numpy()
        print( [encoder.pretty(t) for t in np_inputs[0][:100]] )
        print( {k:v.shape for k,v in batch.items()} )
  
  if os.getenv("SLURM_TMPDIR") is not None:
    # we are on compute canada and should attempt to copy 
    # dataset to tmpdir for faster access
    from shutil import copyfile
    tmpdir = os.getenv("SLURM_TMPDIR")
    dataset_path = os.path.join(tmpdir, os.path.basename(args.dataset))
    if not os.path.exists(dataset_path):
      copyfile(args.dataset, dataset_path)
      copyfile(args.dataset + ".header", dataset_path + ".header")
    args.dataset = dataset_path

  # setup datasets
  train_dataset = dataset_cls(split_id=0, is_training=True, **vars(args))
  eval_dataset = dataset_cls(split_id=2, is_training=False, overload_batches_per_epoch=1, **vars(args))
  Trainer.get_train_dataloader = lambda *_args,**_kwargs: train_dataset
  Trainer.get_eval_dataloader = lambda  *_args,**_kwargs: eval_dataset
  Trainer.compute_loss = loss_fn

  print("MODEL NAME : " + name)
  print("VOCAB SIZE : " + str(vocab_size))
  print("ARGS : " + json.dumps(vars(args),indent=4))
  print("MODEL CONFIG : " + json.dumps(json.load(open(args.config,"r")),indent=4))
  print("ENCODER CONFIG : " + json.dumps(encoder.config.ToJson(),indent=4))

  logging_dir = os.path.join(args.log, "{}".format(name))
  output_dir = os.path.join(args.output, "checkpoints/{}".format(name))

  print("LOGGING PATH : " + logging_dir)
  print("OUTPUT PATH : " + output_dir)
  
  os.makedirs(logging_dir, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)

  # =================================================================
  # model selection

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
    # encoder knows the size of the embedding
    config.n_control_dim = encoder.config.embed_dim 
    model_cls = GPT2LMHeadModelCont
  elif args.arch == "bert":
    config = BertConfig().from_json_file(args.config)
    model_cls = BertForMaskedLM
  else:
    raise NotImplementedError 
  
  config.vocab_size = vocab_size
  print("MODEL CONFIG : " + str(config))

  
  if len(args.ckpt.strip()) == 0:
    print('Model initialization')
    ckpt_path = None
    model = model_cls(config)
  else:
    try:
      print('Trying to load checkpoint')
      ckpt_path = os.path.join(output_dir, f"checkpoint-{args.ckpt_num}")
      model = model_cls.from_pretrained(ckpt_path)
    except Exception as e:
      print(e)
      print('Returning to default model initialization')
      model = model_cls(config)
  

  # Create Memory metrics callback

  # =================================================================
  # training 
  
  training_args = TrainingArguments(
    logging_dir=logging_dir,
    report_to="tensorboard",
    output_dir=output_dir,
    overwrite_output_dir=bool(args.overwrite),
    num_train_epochs=(500000/args.batches_per_epoch)*args.accum_steps,
    logging_steps=args.log_steps,
    save_steps=args.save_steps,
    save_total_limit=None,
    learning_rate=args.lr,
    gradient_accumulation_steps=args.accum_steps,
    per_device_train_batch_size=args.batch_size//args.ngpu//args.accum_steps,
    per_device_eval_batch_size=args.batch_size//args.ngpu//args.accum_steps,
    evaluation_strategy="epoch",
    prediction_loss_only=True,
    skip_memory_metrics=True
  )

  # For custom memory metrics, don't work and multiply by 100 training time!!!
  if args.memory_metrics:
    callbacks = [MemoryUsageCallback, ProfilerCallback]
  else:
    callbacks = []

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=None,
    eval_dataset=None,
    callbacks=callbacks
  )

  trainer.train_dataset = train_dataset
  trainer.eval_dataset = eval_dataset

  if not args.test_only:
    trainer.train(ckpt_path)
  else:
    model = trainer._wrap_model(trainer.model)
    model.save_pretrained(output_dir)
