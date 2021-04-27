# import libraries
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# initialize Lemmatizer instance
lemmatizer = WordNetLemmatizer()


# load intents for chatbot
data_file = open("./resources/intents.json").read()
intents = json.loads(data_file)


# parse intents and get vocabulary and classes
def parse_intents():
    print("Parsing intents to generate vocabulary")
    # words = complete vocabulary
    words = []
    # classes = intents
    classes = []
    # docs = combination between patterns and intents
    docs = []
    ignore_chars = ["?", "!"]
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:

            # tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # add docs in the corpus
            docs.append((w, intent["tag"]))

            # add list of classes
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # lemmatize and lower each word
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word not in ignore_chars
    ]
    # remove duplicates
    words = sorted(list(set(words)))

    # sort classes
    classes = sorted(list(set(classes)))
    return words, classes, docs


# generate training data from parsed intents
def generate_training_data(words, classes, docs):
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in docs:
        # initialize bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word to account for related words
        pattern_words = [
            lemmatizer.lemmatize(word.lower()) for word in pattern_words
        ]
        # create bag of words array, 1 if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle features
    random.shuffle(training)
    training = np.array(training)

    #  X - patterns, Y - intents
    X = list(training[:, 0])
    Y = list(training[:, 1])

    print("Training data created")

    return X, Y


def train_chatbot_model(X, Y):
    # Create sequential model with 3 layers
    model = Sequential()
    # first layer 128 neurons
    model.add(Dense(128, input_shape=(len(X[0]),), activation="relu"))
    model.add(Dropout(0.5))
    # second layer 64 neurons
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    # output layer contains as many neurons as intents to be predicted
    model.add(Dense(len(Y[0]), activation="softmax"))

    # Compile model using Stochastic gradient descent optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
    )

    # fitting data to model
    hist = model.fit(
        np.array(X), np.array(Y), epochs=200, batch_size=5, verbose=1
    )
    # saving the model in local directory
    model.save("./models/chatbot_model.h5", hist)

    print("model created")


vocab, classes, documents = parse_intents()

# save vocab and classes as pickle
pickle.dump(vocab, open("./resources/vocab.pkl", "wb"))
pickle.dump(classes, open("./resources/classes.pkl", "wb"))


train_x, train_y = generate_training_data(vocab, classes, documents)

train_chatbot_model(train_x, train_y)
