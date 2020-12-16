from pickle import load
from tensorflow.keras.models import load_model
import numpy as np

# load a files
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# English
eng_file = 'English_tokens.pkl'
eng_tokens = load_clean_sentences(eng_file)

# German
ger_file = 'German_tokens.pkl'
ger_tokens = load_clean_sentences(ger_file)

# Model
model_file = 'model_4.h5'
model = load_model(model_file, compile=False)


# Input to array
def input_to_array(word):
    c = word.split()

    c = [word.lower() for word in c]
    o = []

    for txt in c:
        word_input = "".join(u for u in txt if u not in ("?", ".", ";", ":", "!"))
        o.append(word_input)

    k = np.zeros(5)
    for i in range(len(o)):
        for key, value in eng_tokens.word_index.items():
            if key == o[i]:
                np.put(k, i, value)
            else:
                pass
    k = k.astype(int)
    k = k.reshape(1, 5)

    return k

# Process Output
def process_output(input_array):
    pred_value = model.predict_classes(input_array)

    vals = []
    for i in pred_value[0]:
        for word, index in ger_tokens.word_index.items():
            if index == i:
                worrdd = word
                vals.append(worrdd)

    output = ' '.join(vals)
    return output

## Flask

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
 

    int_features = [x for x in request.form.values()][0]
    input_array1 = input_to_array(int_features)
    decode_sequence = process_output(input_array1)
    #print(decode_sequence)

    return render_template('index.html', prediction=decode_sequence)

if __name__ == "__main__":
    app.run()
