import pandas as pd
import json
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.utils.np_utils import to_categorical

MAX_NB_WORDS = 5000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 50 # max length of each entry (sentence), including padding
#VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 50      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "~/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

MODEL_PATH = "model.h5"


def save_df_to_file(df, filename):
    dict1 = df.to_dict('records')
    with open(filename, 'wt') as outfp:
        json.dump(dict1, outfp)


def load_dataset(path):
    dataset = {}
    for mode in ['train', 'valid', 'test']:
        df = pd.read_json('{}/{}.json'.format(path, mode))
        dataset[mode] = {'text': df.text.values, 'label': df.label.values}
    return dataset


def load_dataset_as_dataframes(path):
    df = {}
    for mode in ['train', 'valid', 'test']:
        df[mode] = pd.read_json('{}/{}.json'.format(path, mode))
    return df


def create_model(input_length, output_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=input_length))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(output_length, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X, Y):
    
    #X_TRAIN = dataset['train']['text']
    #Y_TRAIN = dataset['train']['label']
    #X_VALID = dataset['valid']['text']
    #Y_VALID = dataset['valid']['label']

    epochs = 40
    batch_size = 64
    history = model.fit(
        X['train'], 
        Y['train'], 
        epochs=epochs, 
        validation_data=(X['valid'], Y['valid']), 
        batch_size=batch_size)


if __name__ == "__main__":

    df = load_dataset_as_dataframes(path='data')

    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    #tokenizer.fit_on_texts(dataset['train']['text'])
    tokenizer.fit_on_texts(df['train'].text)
    word_index = tokenizer.word_index
    print("word_index:", len(word_index))

    X = {}
    Y = {}
    for mode in ['train', 'valid', 'test']:
        #X[mode] = tokenizer.texts_to_sequences(dataset[mode]['text'])
        X[mode] = tokenizer.texts_to_sequences(df[mode].text)
        X[mode] = pad_sequences(X[mode], maxlen=MAX_SEQUENCE_LENGTH)
        Y[mode] = pd.get_dummies(df[mode].label).values
        print("Y[{}] shape: {}".format(mode, Y[mode].shape))

        #x_test = tokenizer.texts_to_sequences(dataset['test']['text'])
        #x_test = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    #tokenized_dataset = {
    #    'train': {'text': x_train, 'label': dataset['train']['label']},
    #    'valid': {'text': x_valid, 'label': dataset['valid']['label']},
    #    'test': {'text': x_test, 'label': dataset['test']['label']}
    #}

    #print(x_train)

    model = create_model(input_length=X['train'].shape[1], output_length=35)
    train_model(model, X, Y)

    model.save(MODEL_PATH)