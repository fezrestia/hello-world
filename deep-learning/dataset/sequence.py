import os
import numpy as np
from pathlib import Path

id_vs_char = {}
char_vs_id = {}

def _update_vocab(text):
    chars = list(text)

    for i, char in enumerate(chars):
        if char not in char_vs_id:
            new_id = len(char_vs_id)
            char_vs_id[char] = new_id
            id_vs_char[new_id] = char

def load_data(file_name = "addition.txt", seed = 1984):
    script_dir = Path(__file__).resolve().parent
    file_path = f"{str(script_dir)}/sequence/{file_name}"

    if not os.path.exists(file_path):
        print(f"file_path = {file_path} is not found.")
        return None

    questions = []
    answers = []

    for line in open(file_path, "r"):
        idx = line.find("_")
        questions.append(line[:idx])  # not include "_"
        answers.append(line[idx:-1])  # include "_"

    # create vocab dict
    for i in range(len(questions)):
        _update_vocab(questions[i])
        _update_vocab(answers[i])

    # create numpy array
    x = np.zeros((len(questions), len(questions[0])), dtype = int)
    t = np.zeros((len(questions), len(answers[0])), dtype = int)

    for i, sentence in enumerate(questions):
        x[i] = [char_vs_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_vs_id[c] for c in list(sentence)]

    # shuffle
    indices = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)

def get_vocab():
    return char_vs_id, id_vs_char

