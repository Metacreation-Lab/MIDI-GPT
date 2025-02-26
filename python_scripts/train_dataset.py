#from transformers import Trainer, TrainingArguments

import os
import json
import time
import torch
import tqdm 
#from torch.utils.data import Dataset

import datetime
import numpy as np

import sys
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
import midigpt

class CustomDataset:
  def __init__(self, split_id=0, is_training=True, batch_size=32, dataset=None, num_bars=4, min_tracks=2, max_tracks=12, max_seq_len=2048, expressive=False, no_max_length=False, resolution=12, encoding=None, pad_value=-100, arch="gpt2", accum_steps=1, batches_per_epoch=1000, overload_batches_per_epoch=None, **kwargs):
    # settings
    self.is_training = is_training
    self.batch_size = batch_size // accum_steps
    self.split_id = split_id
    self.max_seq_len = max_seq_len
    self.batches_per_epoch = batches_per_epoch if overload_batches_per_epoch is None else overload_batches_per_epoch
    self.dataset = list(range(self.batches_per_epoch)) # number of examples ??
    self.pad_value = pad_value
    self.arch = arch

    # create dataloader
    self.dataloader = midigpt.Jagged(dataset)
    self.dataloader.set_num_bars(num_bars)
    self.dataloader.set_min_tracks(min_tracks)
    self.dataloader.set_max_tracks(max_tracks)
    self.dataloader.set_max_seq_len(max_seq_len)
    seed = np.random.randint(2**20)
    self.dataloader.set_seed(seed)
    self.encoder_mode = midigpt.getEncoderType(encoding)
    
    # create train_config
    self.tc = midigpt.TrainConfig()
    self.tc.num_bars = num_bars
    self.tc.min_tracks = min_tracks
    self.tc.max_tracks = max_tracks
    self.tc.use_microtiming = expressive
    self.tc.no_max_length = no_max_length
    self.tc.resolution = resolution

    self.current = 0
  
  def _get_batch(self):
    input_ids, mask = self.dataloader.read_batch_v2(
      self.batch_size, self.split_id, self.encoder_mode, self.tc)
    input_ids = np.array(input_ids)
    mask = np.array(mask)
    labels = np.copy(input_ids)
    labels += (1-mask) * self.pad_value # set masked tokens to pad_value
    batch = {
      "input_ids" : torch.from_numpy(input_ids), 
      "attention_mask" : torch.from_numpy(mask),
      "labels" : torch.from_numpy(labels)
    }
    if self.arch == "xl":
      batch.pop("attention_mask")
      assert np.all(np.sum(mask,axis=1)==self.max_seq_len)
    if self.arch == "bert":
      batch.pop("labels")
    return batch
  
  def _get_batch_test(self):
    inputs = torch.ones((32,800), dtype=torch.int64)
    return {
      "input_ids" : inputs,
      "labels" : inputs
    }

  def __iter__(self):
      self.current = 0
      return self

  def __next__(self):
    self.current += 1
    if self.current <= self.batches_per_epoch:
      while True:
        try:
          return self._get_batch()
        except Exception as e:
          print("ERROR IN BATCHER : ", e)
    raise StopIteration
  
  def __len__(self):
    return self.batches_per_epoch

def pad(seqs, pad_value):
  seqlens = np.array([len(seq) for seq in seqs])
  maxlen = np.max(seqlens)
  return np.array([np.pad(seq, (0,maxlen-len(seq)), mode="constant", constant_values=pad_value) for seq in seqs]), seqlens