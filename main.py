import sys
import json
import numpy as np
from utils import encode_str, load_model, decode_sequence
from typing import Tuple

VOCAB_SIZE = 35
MAX_LEN = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])

    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    # load our idx2word and word2idx mapping
    with open('./weights/idx2word.json', 'r') as file:
        idx2word = json.load(file)
    with open('./weights/word2idx.json', 'r') as file:
        word2idx = json.load(file)
    idx2word = {int(key): str(value) for key, value in idx2word.items()}
    # load pretrain model weight
    f_input = encode_str(factors, word2idx)
    encoder = load_model('./weights/encoder_model.json', './weights/encoder_model_weights.h5')
    decoder = load_model('./weights/decoder_model.json', './weights/decoder_model_weights.h5')

    decoded_sentence = decode_sequence(f_input, encoder, decoder, idx2word, word2idx)
    return decoded_sentence[:-1]


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    # random_list = np.random.choice(np.arange(len(factors)), size=1000)
    # factors = [factors[x] for x in random_list]
    # expansions = [expansions[x] for x in random_list]
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")
