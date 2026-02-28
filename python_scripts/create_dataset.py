import os
import glob
import json
import numpy as np
import csv
import tarfile
import tempfile
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

def worker_bytes(args):
	"""Worker variant that receives raw MIDI bytes instead of a file path (for tar input)."""
	midi_bytes, name, sid, labels, nomml, tcjson, encoding = args
	tc = midigpt.TrainConfig()
	tc.from_json(tcjson)
	labels = dict(labels)
	labels["nomml"] = nomml
	encoder_mode = midigpt.getEncoderType(encoding)
	assert encoder_mode is not midigpt.ENCODER_TYPE.NO_ENCODER
	try:
		with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
			tmp.write(midi_bytes)
			tmp_path = tmp.name
		try:
			return sid, midigpt.midi_to_json_bytes(tmp_path, tc, json.dumps(labels))
		finally:
			os.unlink(tmp_path)
	except Exception as e:
		print(e)
		return None, None

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

def tar_input_generator(tar_path, selected, member_info, tc, encoding):
	"""Lazily stream (midi_bytes, name, sid, labels, nomml, tc, encoding) from a tar archive.

	Reads members sequentially (efficient for tar's sequential format) and only
	extracts the bytes for members that are in `selected`.
	"""
	selected_set = set(selected)
	with tarfile.open(tar_path, 'r:*') as tf:
		for member in tf:
			if member.isfile() and member.name in selected_set:
				fobj = tf.extractfile(member)
				if fobj is not None:
					sid, labels, nomml = member_info[member.name]
					yield fobj.read(), member.name, sid, labels, nomml, tc, encoding

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="")
	parser.add_argument("--tar", type=str, default="",
		help="Path to a .tar archive of MIDI files (alternative to --data_dir). "
		     "Structure-agnostic: all *.mid/*.midi files at any depth are discovered.")
	parser.add_argument("--split", type=str, default="0.8,0.1,0.1",
		help="Train/val/test split ratios used when path markers (-Train-/-Val-/-Test-) "
		     "are absent (e.g. 0.8,0.1,0.1). Only used with --tar.")
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
	parser.add_argument("--metadata", type=str, default="",
		help="Path to metadata CSV with filepath and medianMetricDepth columns. "
		     "Optional when --tar is used.")
	parser.add_argument("--type", type=str, default="Drum+Music")
	parser.add_argument("--test", type=str, default="no")
	args = parser.parse_args()

	if not args.tar and not args.data_dir:
		parser.error("one of --data_dir or --tar is required")

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

	if args.tar:
		# --- Tar-based input path ---
		split_ratios = [float(x) for x in args.split.split(",")]
		if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-6:
			raise ValueError("--split must be three comma-separated ratios summing to 1.0 (e.g. 0.8,0.1,0.1)")

		# Pass 1: scan tar headers only — fast, no data extraction
		print(f"Scanning tar archive: {args.tar}")
		tar_member_names = []
		with tarfile.open(args.tar, 'r:*') as tf:
			for member in tf.getmembers():
				if member.isfile() and member.name.lower().endswith(('.mid', '.midi')):
					tar_member_names.append(member.name)
		print(f"Found {len(tar_member_names)} MIDI files in archive")

		# Determine train(0) / val(1) / test(2) sid for each member.
		# Try path-based markers first; fall back to ratio split if none are found.
		def infer_sid_from_path(name):
			n = name.lower()
			if '-train-' in n: return 0
			if '-val-' in n or '-valid-' in n: return 1
			if '-test-' in n: return 2
			return None

		path_based_sids = [infer_sid_from_path(n) for n in tar_member_names]
		use_ratio_split = all(s is None for s in path_based_sids)

		if use_ratio_split:
			print(f"No split markers found in paths; applying ratio split {split_ratios}")
			random.shuffle(tar_member_names)
			n_total = len(tar_member_names)
			n_train = int(n_total * split_ratios[0])
			n_val = int(n_total * split_ratios[1])
			assigned_sids = [0]*n_train + [1]*n_val + [2]*(n_total - n_train - n_val)
		else:
			assigned_sids = [s if s is not None else 0 for s in path_based_sids]
			# shuffle so max_size trims uniformly
			pairs = list(zip(tar_member_names, assigned_sids))
			random.shuffle(pairs)
			tar_member_names, assigned_sids = zip(*pairs) if pairs else ([], [])
			tar_member_names = list(tar_member_names)
			assigned_sids = list(assigned_sids)

		# Load nomml scores from metadata CSV if provided (matched by file basename)
		nomml_by_basename = {}
		if args.metadata and os.path.exists(args.metadata):
			with open(args.metadata) as meta:
				reader = csv.DictReader(meta, delimiter=',')
				for row in reader:
					try:
						basename = os.path.splitext(os.path.basename(row["filepath"]))[0]
						nomml_by_basename[basename] = int(row["medianMetricDepth"])
					except (KeyError, ValueError):
						pass

		# Load genre/spotify/tension labels
		metadata_label_data = load_metadata_labels(args.genre_data, args.spotify_data, args.tension_data)

		# Build per-member dict: name -> (sid, labels, nomml)
		member_info = {}
		for name, sid in zip(tar_member_names, assigned_sids):
			basename_no_ext = os.path.splitext(os.path.basename(name))[0]
			labels = dict(metadata_label_data.get(basename_no_ext, DEFAULT_LABELS))
			nomml = nomml_by_basename.get(basename_no_ext, 12)
			member_info[name] = (sid, labels, nomml)

		selected = list(tar_member_names)
		if args.max_size > 0:
			selected = selected[:args.max_size]
		print(f"Processing {len(selected)} MIDI files from tar")

		if not test_script:
			total_count = 0
			success_count = 0
			pool = Pool(args.nthreads)
			# Pass 2: stream bytes from tar lazily while workers run in parallel
			gen = tar_input_generator(args.tar, selected, member_info, tc, args.encoding)
			progress_bar = tqdm(pool.imap_unordered(worker_bytes, gen), total=len(selected))
			for sid, b in progress_bar:
				if b is not None and len(b):
					jag.append_bytes_to_file_stream(b, sid)
					success_count += 1
				total_count += 1
				progress_bar.set_description(f"{success_count}/{total_count}")
			jag.close()
		else:
			print("Test successful")
			sys.exit(0)

	else:
		# --- Directory-based input path (original behaviour) ---
		paths = list(glob.glob(args.data_dir + "/**/*.mid", recursive=True))

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
			# Reset to avoid partial state from a failed try block
			paths_all = []
			sids_all = []
			nomml_alls = []

			all_mid_files = list(glob.glob(args.data_dir + "/**/*.mid", recursive=True))
			has_any_marker = any(
				"-train-" in p or "-valid-" in p or "-test-" in p
				for p in all_mid_files
			)

			if has_any_marker:
				for path in all_mid_files:
					if "-train-" in path:
						paths_all.append(path)
						sids_all.append(0)
					elif "-valid-" in path:
						paths_all.append(path)
						sids_all.append(1)
					elif "-test-" in path:
						paths_all.append(path)
						sids_all.append(2)
					else:
						raise RuntimeError("data format incorrect: no split marker in " + path)
				nomml_alls = [12] * len(paths_all)
			else:
				# No path markers — apply ratio split from --split
				dir_split_ratios = [float(x) for x in args.split.split(",")]
				if len(dir_split_ratios) != 3 or abs(sum(dir_split_ratios) - 1.0) > 1e-6:
					raise ValueError("--split must be three comma-separated ratios summing to 1.0")
				random.shuffle(all_mid_files)
				n_total = len(all_mid_files)
				n_train = int(n_total * dir_split_ratios[0])
				n_val = int(n_total * dir_split_ratios[1])
				sids_all = [0]*n_train + [1]*n_val + [2]*(n_total - n_train - n_val)
				random.shuffle(sids_all)  # Interleave so max_size trims uniformly
				paths_all = all_mid_files
				nomml_alls = [12] * n_total
				print(f"No split markers found; applying ratio split {dir_split_ratios}")

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
