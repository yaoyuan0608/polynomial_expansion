import tensorflow as tf
from data_process import load_data, create_training_data
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping 
from keras.losses import CategoricalCrossentropy


BATCH_SIZE = 128
EPOCHS = 10
VOCAB_SIZE = 35
MAX_LEN = 29
EMBEDDING_DIM = 64

def loss_func(targets, pred):
    cross_entropy = CategoricalCrossentropy()
    mask = tf.math.logical_not(tf.math.equal(tf.math.argmax(pred,axis=2), 0))
    mask = tf.expand_dims(mask, axis=-1)
    loss = cross_entropy(targets, pred, sample_weight=mask)
    return loss


filepath = 'train.txt'

if __name__ == '__main__':

    print('Loading data.....')
    train, label = load_data(filepath)
    encoder_input_data, decoder_input_data, decoder_target_data = create_training_data(train, label)

    #define an input of the encoder with length as the number of encoder tokens
    encoder_inputs = Input(shape=(None, VOCAB_SIZE))
    encoder = Bidirectional(LSTM(EMBEDDING_DIM, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, VOCAB_SIZE))    
    decoder_lstm = LSTM(EMBEDDING_DIM*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #Train the model
    model.compile(optimizer='rmsprop', loss=loss_func, metrics=['accuracy'])
    callback = EarlyStopping(monitor='loss', patience=1)

    print('Start training....')
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             validation_split=0.2, verbose=2, callbacks=[callback])

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(EMBEDDING_DIM*2,))
    decoder_state_input_c = Input(shape=(EMBEDDING_DIM*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    with open('./weights/encoder_model.json', 'w', encoding='utf8') as f:
        f.write(encoder_model.to_json())
    encoder_model.save_weights('./weights/encoder_model_weights.h5')

    with open('./weights/decoder_model.json', 'w', encoding='utf8') as f:
        f.write(decoder_model.to_json())
    decoder_model.save_weights('./weights/decoder_model_weights.h5')

    print('Finish training...')