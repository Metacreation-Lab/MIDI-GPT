# take a trained pytorch model and convert it
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
print( os.path.dirname(os.getcwd()) + "/python_lib" )
import midigpt
import time
import json
import numpy as np
import torch
import torch.quantization
from transformers import GPT2LMHeadModel,  GPT2Config
from transformers.modeling_utils import Conv1D

from custom_models import *

from torch import nn

class QuantWrapper(nn.Module):
    def __init__(self, module):
        super(QuantWrapper, self).__init__()
        qconfig = module.qconfig if hasattr(module, 'qconfig') else None
        self.add_module('quant', torch.quantization.QuantStub(qconfig))
        self.add_module('dequant', torch.quantization.DeQuantStub())
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X, P):
        X = self.quant(X)
        P = self.quant(P)
        O = self.module(X,P)
        return self.dequant(O)

def _conv1d_to_linear(module):
  in_size, out_size = module.weight.shape
  linear = torch.nn.Linear(in_size, out_size)
  linear.weight.data = module.weight.data.T.contiguous()
  linear.bias.data = module.bias.data
  return linear

def conv1d_to_linear(model):
  for name in list(model._modules):
    module = model._modules[name]
    if isinstance(module, Conv1D):
      linear = _conv1d_to_linear(module)
      model._modules[name] = linear
    else:
      conv1d_to_linear(module)

def score_model(model):
  targets = np.load("target.npz")["data"]
  

def time_model(model):
  start = time.time()
  pkv = None
  for _ in range(1000):
    input_ids = torch.ones(1,1).type(torch.LongTensor)
    outputs = model(input_ids, past_key_values=pkv)
    pkv = outputs[1]
  print("BATCH TIME : {}".format(time.time() - start))

def print_size_of_model(model):
  import os
  torch.save(model.state_dict(), "temp.p")
  print('Size (MB):', os.path.getsize("temp.p")/1e6)
  os.remove('temp.p')

def quantize_model(model):
  conv1d_to_linear(model)
  model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8)
  return model

def static_quantize_model(model):
  conv1d_to_linear(model)
  model.qconfig = torch.quantization.default_qconfig
  torch.quantization.prepare(model, inplace=True)
  torch.quantization.convert(model, inplace=True)
  
  return model

def prune_model(model):
  import torch.nn.utils.prune as prune

  conv1d_to_linear(model)
  parameters_to_prune = []
  for _,module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
      prune.l1_unstructured(module, name="weight", amount=.8)
      prune.remove(module, "weight")

    for _,submodule in module.named_modules():
      if isinstance(submodule, torch.nn.Linear):
        prune.l1_unstructured(submodule, name="weight", amount=.8)
        prune.remove(submodule, "weight")

  return model

def inject_metadata(path, metadata_path, encoder, new_state):
  model = torch.jit.load(path)
  with open(metadata_path, "r") as f:
    metadata = json.load(f)
  metadata["encoder"] = encoder
  metadata["new_state"] = new_state
  extra_files = torch._C.ExtraFilesMap()
  extra_files['metadata.json'] = json.dumps(metadata)
  out_path = os.path.splitext(path)[0] + "_WMETA.pt"
  torch.jit.save(model, out_path, _extra_files=extra_files)

def convert(model, path, quantize=False, prune=False, force=False, control=False, ckpt_path=None, encoderX=None):
  if not os.path.exists(path) or force:
    model.eval()
    if quantize:
      model = quantize_model(model)
    if prune:
      model = prune_model(model)
    print_size_of_model(model)    
    example_input = torch.zeros(1,300).type(torch.LongTensor)
    example_control = torch.zeros(1,300,3).type(torch.FloatTensor)
    if control:
      outputs = model(input_ids=example_input, control_ids=example_control, past_key_values=None)
      print(len(outputs[1]))
      traced_script_module = torch.jit.trace(model, [example_input,example_control,outputs[1]])
    else:
      outputs = model(input_ids=example_input)
      traced_script_module = torch.jit.trace(model, [example_input, outputs[1]])
    
    num_layers = len(outputs[1])
    _,num_heads,_,num_hidden = outputs[1][0][0].detach().numpy().shape
    encoder = encoderX

    model_metadata = {
      "encoder" : encoder,
      "num_heads" : num_heads,
      "num_hidden" : num_hidden,
      "num_layers" : num_layers,
      "model_dim" : -1,
      "new_state" : True
    }

    print(model_metadata)
    
    extra_files = {}
    extra_files['metadata.json'] = json.dumps(model_metadata)
    torch.jit.save(
      traced_script_module, path, _extra_files=extra_files)


class GPT2LMHeadModelWMeta(GPT2LMHeadModel):
  def extra_repr(self):
    return "trent is the man"

if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt_path", type=str, required=True)
  parser.add_argument("--output", type=str, default="")
  parser.add_argument("--metadata_path", type=str, default="")
  parser.add_argument("--config", type=str, default="")
  parser.add_argument("--encoder", type=str, default="NONE")
  parser.add_argument("--init", action="store_true")
  parser.add_argument("--inject", action="store_true")
  parser.add_argument("--new_state", action="store_true")
  parser.add_argument("--quantize", action="store_true")
  parser.add_argument("--prune", action="store_true")
  parser.add_argument("--control", action="store_true")

  args = parser.parse_args()


  if args.inject:
    assert len(args.metadata_path)
    inject_metadata(
      args.ckpt_path, args.metadata_path, args.encoder, True if args.new_state else False)
  
  else:
    assert len(args.output)
    if args.init:
      encoder_mode = midigpt.getEncoderType(args.encoder)
      assert encoder_mode is not midigpt.ENCODER_TYPE.NO_ENCODER
      encoder = midigpt.getEncoder(encoder_mode)
      vocab_size = encoder.vocab_size()
      if args.control:
        config = GPT2LMHeadModelContConfig().from_json_file(args.config)
        # encoder knows the size of the embedding
        config.n_control_dim = encoder.config.embed_dim 
        model_cls = GPT2LMHeadModelCont
        
      else:
        config = GPT2Config().from_json_file(args.config)
        config.vocab_size = vocab_size
        model_cls = GPT2LMHeadModel

      model = model_cls(config)
    else:
      if args.control:
        model = GPT2LMHeadModelCont.from_pretrained(args.ckpt_path, torchscript=True)
      else:
        model = GPT2LMHeadModel.from_pretrained(args.ckpt_path, torchscript=True)
    
    convert(model, args.output, quantize=args.quantize, prune=args.prune, control=args.control, ckpt_path=args.ckpt_path, encoderX=args.encoder)

