from os import listdir
from os.path import isfile, join
from shutil import move
import random
import sys
from tqdm import tqdm

if __name__ == "__main__":

    root = sys.argv[1]
    new_root = sys.argv[2]

    # Get all midi files
    print("Getting MIDI files")
    onlyfiles = [f for f in listdir(root)]
    n = len(onlyfiles)
    print("Num files: " + str(n))

    # Generate random test/train/valid indices

    idx = [i for i in range(n)]
    split_idx = random.shuffle(idx)

    train_len = int(0.8 * n)
    test_len = int(0.1 * n)
    valid_len = n - train_len - test_len

    train_idx = idx[:train_len]
    test_idx = idx[train_len:test_len + train_len]
    valid_idx = idx[test_len + train_len:]

    # Move files to respective folder

    o = 0

    print('Spliting Train Set')
    for i in tqdm(range(train_len)):
        move(join(root, onlyfiles[train_idx[i]]), join(new_root, "train", onlyfiles[train_idx[i]]))
        o += 1

    print('Spliting Test Set')
    for i in tqdm(range(test_len)):
        move(join(root, onlyfiles[test_idx[i]]), join(new_root, "test", onlyfiles[test_idx[i]]))
        o += 1

    print('Spliting Validation Set')
    for i in tqdm(range(valid_len)):
        move(join(root, onlyfiles[valid_idx[i]]), join(new_root, "valid", onlyfiles[valid_idx[i]]))
        o += 1

    print("Succes Test: " + str(o == n))

    print(join(new_root, "valid", onlyfiles[valid_idx[100]]))
    print(isfile(join(new_root, "valid", onlyfiles[valid_idx[100]])))