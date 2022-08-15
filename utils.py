import re
import numpy as np
from keras.models import model_from_json

VOCAB_SIZE = 35
MAX_LEN = 29


def reg_sen(x):
    return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", x.strip().lower())


def parse_embedding(f):
    encoder_input_data = np.zeros((1, MAX_LEN, VOCAB_SIZE), dtype='float32')
    for t, char in enumerate(f):
        encoder_input_data[0, t, char] = 1.
    return encoder_input_data


def encode_str(f, word2idx):
    f = reg_sen(f)
    f_seq = [word2idx[i] for i in f]
    f_pad = f_seq + [0] * (MAX_LEN - len(f_seq))
    f_input = parse_embedding(f_pad)
    return f_input


def load_model(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model


def decode_sequence(input_seq, encoder, decoder, idx2word, word2idx):
    states_value = encoder.predict(input_seq)
    target_seq = np.zeros((1, 1, VOCAB_SIZE))
    target_seq[0, 0, word2idx['#']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx2word[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '$' or len(decoded_sentence) > MAX_LEN):
            stop_condition = True

        target_seq = np.zeros((1, 1, VOCAB_SIZE))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]


    return decoded_sentence
