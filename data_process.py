import re
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def reg_sen(x):
    return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", x.strip().lower())

def text2seq(train, label, VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, split=' ', filters='')
    tokenizer.fit_on_texts(train + label)
    encode_seq = tokenizer.texts_to_sequences(train)
    decode_seq = tokenizer.texts_to_sequences(label)
    dictionary = tokenizer.word_index
    
    word2idx = {}
    idx2word = {}
    for k, v in dictionary.items():
        if v < VOCAB_SIZE:
            word2idx[k] = v
            idx2word[v] = k
        if v >= VOCAB_SIZE-1:
            continue
          
    return word2idx, idx2word, encode_seq, decode_seq

def padding(seq, MAX_LEN):
    encode_seq = pad_sequences(seq, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    return encode_seq

def load_data(filepath):
    # train and label examples:
    # ['( 7 - 3 * z ) * ( - 5 * z - 9 )', '- 9 * s ** 2', '( 2 - 2 * n ) * ( n - 1 )', 'x ** 2', '( 4 - x ) * ( x - 2 3 )']
    # ['# 1 5 * z ** 2 - 8 * z - 6 3 $', '# - 9 * s ** 2 $', '# - 2 * n ** 2 + 4 * n - 2 $', '# x ** 2 $', '# - x ** 2 + 2 7 * x - 9 2 $']
    with open(filepath) as file:
        raw = file.readlines()

    train = []
    label = []
    for row in raw:
        row = row.strip('\n')
        _train, _label = row.split('=')
        _train = reg_sen(_train)
        _label = reg_sen(_label)
        _label = ['#'] + _label + ['$']
        train.append(' '.join(_train))
        label.append(' '.join(_label))
    return train, label

def create_training_data(train, label):
    train_unique = len(set(' '.join(train).split(' ')))
    label_unique = len(set(' '.join(label).split(' ')))

    MAX_LEN = 29
    VOCAB_SIZE = max(train_unique, label_unique)+1
    # print(VOCAB_SIZE) -> 35
    EMBEDDING_DIM = 64
    word2idx, idx2word, encode_seq, decode_seq = text2seq(train, label, VOCAB_SIZE)

    word2idx[' '] = 0
    idx2word[0] = ' '

    with open('./weights/word2idx.json', 'w') as file:
        json.dump(word2idx, file)
    with open('./weights/idx2word.json', 'w') as file:
        json.dump(idx2word, file)

    encode_input = padding(encode_seq, MAX_LEN)
    decode_input = padding(decode_seq, MAX_LEN)

    encoder_input_data = np.zeros((len(train), MAX_LEN, VOCAB_SIZE), dtype='float32')
    decoder_input_data = np.zeros((len(train), MAX_LEN, VOCAB_SIZE), dtype='float32')
    decoder_target_data = np.zeros((len(label), MAX_LEN, VOCAB_SIZE), dtype='float32')

    #parse the input and output texts (one-hot)
    for i, (_encode, _decode) in enumerate(zip(encode_input, decode_input)):
        for t, char in enumerate(_encode):
            encoder_input_data[i, t, char] = 1.
        for t, char in enumerate(_decode):
            decoder_input_data[i, t, char] = 1.
            if t > 0:
                decoder_target_data[i, t-1, char] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data