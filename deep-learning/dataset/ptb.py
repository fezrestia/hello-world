import sys
import os
import importlib
from pathlib import Path
import pickle
import numpy as np

def reload():
    importlib.reload(sys.modules[__name__])

# data_type : one of train, test or valid
def load_data(data_type = "train"):
    script_dir = Path(__file__).resolve().parent
    data_dir = str(script_dir) + "/ptb"

    # load vocab data.
    pkl_file = data_dir + "/ptb.vocab.pkl"
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            word_vs_id, id_vs_word = pickle.load(f)
    else:
        word_vs_id = {}
        id_vs_word = {}

        data_file = data_dir + f"/ptb.train.txt"

        words = open(data_file).read().replace('\n', '<eos>').strip().split()

        for i, word in enumerate(words):
            if word not in word_vs_id:
                word_id = len(word_vs_id)
                word_vs_id[word] = word_id
                id_vs_word[word_id] = word

        with open(pkl_file, "wb") as f:
            pickle.dump((word_vs_id, id_vs_word), f)

    # load corpus data.
    corpus_file = data_dir + f"/ptb.corpus.{data_type}.npy"
    if os.path.exists(corpus_file):
        corpus = np.load(corpus_file)
    else:
        data_file = data_dir + f"/ptb.{data_type}.txt"

        words = open(data_file).read().replace('\n', '<eos>').strip().split()

        corpus = np.array([word_vs_id[w] for w in words])

        np.save(corpus_file, corpus)

    return corpus, word_vs_id, id_vs_word

