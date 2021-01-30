import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
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
    model.add(tf.keras.layers.LSTM(110, dropout=0.2, recurrent_dropout=0.2))
    #model.add(tf.keras.layers.Dense(300, activation='relu'))
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
    label_names = {}
    for mode in ['train', 'valid', 'test']:
        #X[mode] = tokenizer.texts_to_sequences(dataset[mode]['text'])
        X[mode] = tokenizer.texts_to_sequences(df[mode].text)
        X[mode] = pad_sequences(X[mode], maxlen=MAX_SEQUENCE_LENGTH)
        y_encode = pd.get_dummies(df[mode].label)
        label_names[mode] = list(y_encode.columns)
        Y[mode] = y_encode.values
        print("{}: {}".format(mode, label_names[mode]))
        print("Y[{}] shape: {}".format(mode, Y[mode].shape))

        #x_test = tokenizer.texts_to_sequences(dataset['test']['text'])
        #x_test = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    #tokenized_dataset = {
    #    'train': {'text': x_train, 'label': dataset['train']['label']},
    #    'valid': {'text': x_valid, 'label': dataset['valid']['label']},
    #    'test': {'text': x_test, 'label': dataset['test']['label']}
    #}

    #print(x_train)

    MODE = "train"

    if MODE == "train":
        model = create_model(input_length=X['train'].shape[1], output_length=35)
        train_model(model, X, Y)
        model.save(MODEL_PATH)

    elif MODE == "test": # test saved model
        model = load_model(MODEL_PATH)


    print(model.summary())

    # Check on test data

    predictions = model.predict(X['test'])
    print("predictions:", predictions)
    print("predictions.shape:", predictions.shape)
    print("Y_test:", Y['test'])
    print("Y_test.shape:", Y['test'].shape)

    convert_to_index = lambda arr: np.apply_along_axis(np.argmax, 1, arr)
    predicted_index = convert_to_index(predictions)
    test_index = convert_to_index(Y['test'])

    print(predicted_index)
    print(test_index)
    test_acc = np.sum(predicted_index == test_index) / len(predicted_index)
    print("test_acc = {}".format(test_acc)) # test_acc = 0.967799



    # Test on samples
    print("\n\n Test on examples")

    samples = [
       ("Navigate to url www.hi.net/#/login .", "OPEN_WEBSITE"),

       ("Fill in text in BOQ Street Address line 1 EOQ .", "ENTER"),
       ('Set text in XPATH .', "ENTER"),

       ("choose XPATH New PAN Indian Citizen Form 49A .", "SELECT"),
       ('select BOQ Type EOQ .', "SELECT"),
       ('select Type .', "SELECT"),
       ('select .', "SELECT"),
       ('select BOQ Partnership Firm EOQ in BOQ PAN_APPLCNT_STATUS EOQ .', "SELECT"),

       ("Click XPATH .", "CLICK"),
       ("Press XPATH .",  "CLICK"),
       ('click BOQ Log In EOQ .',  "CLICK"),
       ('Click on first BOQ MoreVert EOQ .', "CLICK"),

       ("assert password after username .", "VERIFY"),
       ("assert username .", "VERIFY"),
       ('Verify text BOQ Disable Test Case EOQ .', "VERIFY"),

       ("Verify XPATH width is BOQ 235px EOQ .", "VERIFY_CSSPROP"),

       ("Verify XPATH is enabled .", "VERIFY_XPATH"),
       ("Verify XPATH is google .", "VERIFY_XPATH"),
       ('Verify XPATH contains Current or contains events .', "VERIFY_XPATH"),
       ('Verify XPATH contains Current or ends with events .', "VERIFY_XPATH"),
       ('Verify XPATH is Related Changes .', "VERIFY_XPATH"),
       ('verify XPATH is ername .', "VERIFY_XPATH"),
       
       ('scroll down .', "SCROLL_ACTION"),
       ('scroll up .', "SCROLL_ACTION"),
       
       ('Hit Enter .', "HIT"),
       ('Hit escape .', "HIT"),
       ('Hit spacebar .', "HIT"),
       ('hit tab .', "HIT"),
       ('hit up arrow key .', "HIT"),
       ('Begin block Block1 .', "BEGIN"),

       ('verify BOQ New Quote EOQ is visible on the page .', "VERIFY"),
       ('verify BOQ INSIDEQOUTES1 EOQ is visible on the page .', "VERIFY"),
    ]


    input_texts = list(map(lambda x: x[0], samples))
    true_labels = list(map(lambda x: x[1], samples))

    tokenized_texts = tokenizer.texts_to_sequences(input_texts)
    tokenized_texts = pad_sequences(tokenized_texts, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(tokenized_texts)
    predicted_index = convert_to_index(predictions)
    predicted_labels = list(map(lambda index: label_names['train'][index], predicted_index))

    count = 0
    for i in range(len(samples)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        if true_label == predicted_label:
            print("+ {}".format(predicted_label))
            count += 1
        else:
            print("- wrong prediction: true={}, predicted={}".format(true_label, predicted_label))

    total_count = len(samples)
    acc = count / total_count
    print("\nacc = {:.4f}  [{}/{}]".format(acc, count, total_count))    