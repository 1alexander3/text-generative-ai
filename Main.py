# Import libraries
import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

# Import TensorFlow 
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

# Save csv file to a variable, tranform it into a list, join the text, and limit the characters
text_df = pd.read_csv("[Enter CSV file here]")
text = list(text_df.text.values)
joined_text = " ".join(text)
partial_text = joined_text[:10000]

# Tokenize the individual words and lowercase partial text
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

# Prevent duplicate tokens and index each token
unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}

# Number of words and structure for X and y
n_words = 10
input_words = []
next_words = []

# Predicting the words after number of n_words
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

# Turn input_words and next_words into numpy arrays X and y respectively
X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)

# Go through all words in input_words, map the specific word in X, and map it to the index
for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1

# Creation of model
model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))

# Compile and adjust training/epoch size for model
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
model.fit(X, y, batch_size=128, epochs=30, shuffle=True)

# Save the model and load it
model.save("mymodel.keras")
model = load_model("mymodel.keras")

# Function to create predictive text based on input and the number of best predictions
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1
    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

# Testing possible next words, taking the index and getting their respective tokens
possible = predict_next_word("He will have to look into this thing and he", 5)
print([unique_tokens[idx] for idx in possible])

# Function that will take input, length of text, and creativity that will determine how many words to choose from
def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)

# Testing generative text
print(generate_text("He will have to look into this thing and he", 100, 5))
