

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model, Sequential

df = pd.read_csv(r'data\deu.txt',sep = '\t')

df.drop_duplicates(subset='1', keep=False, inplace=True)

data = np.array(df[:12000])

# Remove Punctuations and Lower Case

def processing(data):
    clean = []
    for words in data:
        # Remove Punctuation
        cleaned_word = "".join(letter for letter in words if letter not in ("?", ".", ";", ":", "!"))
        clean.append(cleaned_word)
    # Lower Case
    clean = [word.lower() for word in clean]
    return clean

data[:,0] = processing(data[:,0])
data[:,1]= processing(data[:,1])

# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# sentence length
def vocab_length(lines):
    lengths = [len(line.split()) for line in lines]
    return lengths

eng_vocabs = vocab_length(data[:, 0])
ger_vocabs = vocab_length(data[:, 1])

# prepare english tokenizer
eng_tokenizer = tokenization(data[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max(eng_vocabs)

print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

# prepare Deutch tokenizer
ger_tokenizer = tokenization(data[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max(ger_vocabs)

print('German Vocabulary Size: ', ger_vocab_size)
print('German Max Length: ', ger_length)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state = 42)

# prepare training data
trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_output(trainY, ger_vocab_size)

# prepare validation data
testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_output(testY, ger_vocab_size)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
      model = Sequential()
      model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
      model.add(LSTM(n_units))
      model.add(RepeatVector(tar_timesteps))
      model.add(LSTM(n_units, return_sequences=True))
      model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
      return model

# define model
model = define_model(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 250)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

filename = 'model1.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=68, validation_data=(testX, testY), callbacks=[checkpoint], verbose=1)


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

loss = model.history.history['loss']
accuracy = model.history.history['loss']
val_loss = model.history.history['val_loss']
val_accuracy = model.history.history['val_accuracy']

dump(loss, open('data/loss.pkl', 'wb'))
dump(loss, open('data/accuracy.pkl', 'wb'))
dump(loss, open('data/val_loss.pkl', 'wb'))
dump(loss, open('data/val_accuracy.pkl', 'wb'))


def input_to_array(word):
    c = word.split()

    c = [word.lower() for word in c]
    o = []

    for txt in c:
        word_input = "".join(u for u in txt if u not in ("?", ".", ";", ":", "!"))
        o.append(word_input)

    k = np.zeros(5)
    for i in range(len(o)):
        for key, value in eng_tokenizer.word_index.items():
            if key == o[i]:
                np.put(k, i, value)
            else:
                pass
    k = k.astype(int)
    k = k.reshape(1, 5)

    return k


def process_output(input_array, tokenizer):
    pred_value = model.predict_classes(input_array)

    vals = []
    for i in pred_value[0]:
        for word, index in tokenizer.word_index.items():
            if index == i:
                worrdd = word
                vals.append(worrdd)

    output = ' '.join(vals)
    return output

flag = True

while(flag == True):
    input_english = input('Enter a text in English : ')
    if( input_english != '123'):
        input_array1 = input_to_array(input_english)
        decode_sequence = process_output(input_array1, ger_tokenizer)
        print("Output French : ", decode_sequence)
    else:
        flag = False