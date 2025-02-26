import json
import jsonlines
from multiprocessing import Pool
from tqdm import tqdm

def load_json(path):
	with open(path,"r") as f:
		return json.load(f)

def dump_json(x, path):
	with open(path,"w") as f:
		json.dump(x,f,indent=4)

def apply_func(x, func):
  pool = Pool(8)
  for inval,outval in tqdm(pool.imap_unordered(func,x),total=len(x)):
    yield inval,outval

def load_jsonl(path,max_items=None):
	with jsonlines.open(path) as reader:
		for ii,item in enumerate(reader):
			yield item
			if max_items is not None and ii >= max_items:
				break

def dump_jsonl(data, path):
	assert isinstance(data, list)
	with jsonlines.open(path, mode="w") as wr:
		for item in tqdm(data,leave=False):
			wr.write(item)

class dump_jsonl_multistage:
	def __init__(self, path, mode="a"):
		self.wr = jsonlines.open(path, mode=mode, flush=True)
	def add(self, item):
		self.wr.write(item)
	def extend(self, items):
		for item in items:
			self.add(item)
	def close(self):
		self.wr.close()
