#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plot

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from network.DeepLSTMLM import DeepLSTMLM

class DeepLSTMLMGen(DeepLSTMLM):
    def generate(self, start_id, skip_ids = None, sample_size = 100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)  # 1 batch, 1 word
            score = self.predict(x)

            p = resource.softmax_func(score.flatten())  # 1 batch, V word score (1, V) -> flatten to (V, )

            sampled = np.random.choice(len(p), size = 1, p = p)  # sampled : (1, )

            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x[0]))  # x : (1, ) -> 1

        return word_ids



# RUN
if __name__ == "__main__":
    import dataset.ptb as ptb

    corpus, word_vs_id, id_vs_word = ptb.load_data("train")
    vocab_size = len(word_vs_id)
    corpus_size = len(corpus)

    model = DeepLSTMLMGen()

    script_dir = Path(__file__).resolve().parent
    data_dir = str(script_dir) + "/../dataset/ptb"
    pkl_file = f"{data_dir}/lstmlm_params.pkl"
    model.load_params(pkl_file)

    start_word = "you"
    start_id = word_vs_id[start_word]
    skip_words = [
            "N",
            "<unk>",
            "$",
    ]
    skip_ids = [word_vs_id[w] for w in skip_words]

    word_ids = model.generate(start_id, skip_ids)
    text = " ".join([id_vs_word[i] for i in word_ids])
    text = text.replace(" <eos>", ".\n")

    print(f"{text}")


    model.reset_state()

    start_words = "the meaning of life is"
    start_ids = [word_vs_id[w] for w in start_words.split(" ")]

    for x in start_ids[:-1]:
        x = np.array(x).reshape(1, 1)
        model.predict(x)

    word_ids = model.generate(start_ids[-1], skip_ids)

    word_ids = start_ids[:-1] + word_ids
    text = " ".join([id_vs_word[i] for i in word_ids])
    text = text.replace(" <eos>", ".\n")

    print("-" * 64)
    print(f"{text}")

