import os
import glob
import json
import numpy as np
import csv
from tqdm import tqdm
from multiprocessing import Pool

from utils import *

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()) + "/python_lib")
import midigpt

def worker(args):
	path,sid,labels,nomml,tcjson,encoding = args
	tc = midigpt.TrainConfig()
	tc.from_json(tcjson)
	labels["nomml"] = nomml

	encoder_mode = midigpt.getEncoderType(encoding)
	assert encoder_mode is not midigpt.ENCODER_TYPE.NO_ENCODER
	encoder = midigpt.getEncoder(encoder_mode)

	try:
		return sid, midigpt.midi_to_json_bytes(path,tc,json.dumps(labels))
	except Exception as e:
		print(e)
		return None,None

def load_json(path):
	if not os.path.exists(path):
		return {}
	with open(path, "r") as f:
		return json.load(f)

DEFAULT_LABELS = {
	"genre": "GENRE_MUSICMAP_ANY",
	"valence_spotify": -1,
	"energy_spotify": -1,
	"danceability_spotify": -1,
	"tension": []
}

DATA_TYPES = [
	"Drum",
	"Drum+Music",
	"Music-No-Drum"
]

def load_metadata_labels(genre_data_path, spotify_data_path, tension_data_path):
	data = {}
	genre_data = load_json(genre_data_path)
	spotify_data = load_json(spotify_data_path)
	tension_data = load_json(tension_data_path)
	md5s = list(set(list(genre_data.keys()) + list(spotify_data.keys()) + list(tension_data.keys())))
	for md5 in md5s:
		data[md5] = {}
		if md5 in spotify_data:
			data[md5]["valence_spotify"] = np.mean(spotify_data[md5]["valence"])
			data[md5]["energy_spotify"] = np.mean(spotify_data[md5]["energy"])
			data[md5]["danceability_spotify"] = np.mean(spotify_data[md5]["danceability"])
		else:
			for k,v in DEFAULT_LABELS.items():
				data[md5][k] = v
		data[md5]["genre"] = genre_data.get(md5, DEFAULT_LABELS["genre"])
		data[md5]["tension"] = tension_data.get(md5, DEFAULT_LABELS["tension"])
	return data

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--num_bars", type=int, default=4)
	parser.add_argument("--expressive", action="store_true")
	parser.add_argument("--ignore_score", type=bool, default=0)
	parser.add_argument("--nthreads", type=int, default=8)
	parser.add_argument("--max_size", type=int, default=-1)
	parser.add_argument("--genre_data", type=str, default="")
	parser.add_argument("--spotify_data", type=str, default="")
	parser.add_argument("--tension_data", type=str, default="")
	parser.add_argument("--encoding", type=str, default="TRACK_ENCODER")
	parser.add_argument("--resolution", type=int, default=12)
	parser.add_argument("--delta_resolution", type=int, default=1920)
	parser.add_argument("--metadata", type=str, required=True)
	parser.add_argument("--type", type=str, default="Drum+Music")
	parser.add_argument("--test", type=str, default="no")
	args = parser.parse_args()

	args.ignore_score = bool(args.ignore_score)
	if args.test != "no":
		test_script = True
	else:
		test_script = False

	assert args.type in DATA_TYPES
	args.type = "-" + args.type + "-"

	import os
	os.system("taskset -p 0xffff %d" % os.getpid())

	# multi thread approach takes about 2 minutes
	pool = Pool(args.nthreads)
	output = os.path.splitext(args.output)[0]
	ss=""
	if args.max_size > 0:
		ss=f"_MAX_{args.max_size}"
	if args.expressive:
		output += "/{}_NUM_BARS={}_RESOLUTION_{}_DELTA_{}{}.arr".format(args.encoding,args.num_bars,args.resolution, args.delta_resolution,ss)
	else:
		output += "/{}_NUM_BARS={}_RESOLUTION_{}{}.arr".format(args.encoding,args.num_bars,args.resolution,ss)
	print(output)
	if not test_script:
		jag = midigpt.BytesToFile(output)
	

	paths = list(glob.glob(args.data_dir + "/**/*.mid", recursive=True))
	
	import random
	import time
	random.seed(int(time.time()))

	tc = midigpt.TrainConfig()
	tc.num_bars = args.num_bars
	tc.use_microtiming = args.expressive
	tc.resolution = args.resolution
	tc.delta_resolution = args.delta_resolution
	tc = tc.to_json()
	print(tc)
	
	paths_exp = []
	sids_exp = []
	paths_non_exp = []
	sids_non_exp = []
	paths_all = []
	sids_all = []
	nomml_alls = []
	nomml_scores = []

	try:
		with open(args.metadata) as meta:
			reader = csv.DictReader(meta, delimiter=',')
			for row in tqdm(reader):
				path = row["filepath"]
				nomml = int(row["medianMetricDepth"])
				if (".mid" in path and args.type in path):
					if "-Train-" in path:
						group = 0
					elif "-Val-" in path:
						group = 1
					elif "-Test-" in path:
						group = 2
					else:
						raise RuntimeError("data format incorrect")
					if (nomml < 12):
						paths_non_exp.append(os.path.join(args.data_dir,path))
						sids_non_exp.append(group)
						nomml_scores.append(nomml)
					else:
						paths_exp.append(os.path.join(args.data_dir,path))
						sids_exp.append(group)
					paths_all.append(os.path.join(args.data_dir,path))
					sids_all.append(group)
					nomml_alls.append(nomml)
		
	except:
		paths_all = list(glob.glob(args.data_dir + "/**/*.mid", recursive=True))
		for path in paths_all:
			if "-train-" in path:
				sids_all.append(0)
			elif "-valid-" in path:
				sids_all.append(1)
			elif "-test-" in path:
				sids_all.append(2)
			else:
				raise RuntimeError("data format incorrect")

	nomml_vals = []
	if args.expressive:
		if args.ignore_score:
			paths = paths_exp
			sids = sids_exp
			nomml_vals = [12 for _ in sids]
		else:
			paths = paths_all
			sids = sids_all
			nomml_vals = nomml_alls
	else:
		paths = paths_all
		sids = sids_all
		nomml_vals = nomml_alls

	metadata_label_data = load_metadata_labels(args.genre_data, args.spotify_data, args.tension_data)
	metadata_labels = [metadata_label_data.get(os.path.splitext(os.path.basename(p))[0],DEFAULT_LABELS) for p in paths]
	print("LOADED {} METADATA LABELS".format(len(metadata_labels)))

	tcs = [tc for _ in paths]
	encoding = [args.encoding for _ in paths]
	inputs = list(zip(paths,sids,metadata_labels,nomml_vals,tcs,encoding))
	random.shuffle(inputs)

	for k,v in DEFAULT_LABELS.items():
		print("{} FILES HAVE {} METADATA".format(sum([m[k] != v for m in metadata_labels]),k))

	if args.max_size > 0:
		inputs = inputs[:args.max_size]

	if not test_script:
		total_count = 0
		success_count = 0
		pool = Pool(args.nthreads)
		progress_bar = tqdm(pool.imap_unordered(worker, inputs), total=len(inputs))
		for sid,b in progress_bar:
			if b is not None and len(b):
				jag.append_bytes_to_file_stream(b,sid)
				success_count += 1
			total_count += 1
			status_str = "{}/{}".format(success_count,total_count)
			progress_bar.set_description(status_str)
		jag.close()
	else:
		print("Test successful")
		sys.exit(0)
