# import libraries
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import numpy as np

# options for language detection models
lang_detect_model = ["googletrans", "fasttext"]

# initialize Lemmatizer instance
lemmatizer = WordNetLemmatizer()

intents = json.loads(open("./resources/intents.json").read())
vocabulary = pickle.load(open("./resources/vocab.pkl", "rb"))
classes = pickle.load(open("./resources/classes.pkl", "rb"))

model = load_model("./models/chatbot_model.h5")

# check if punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# check if wordnet corpora is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


# preprocess user input text
def prepocess_text(text):
    # tokenize the text
    words = nltk.word_tokenize(text)
    # lemmatize each word
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words


# return bag of words array: 0 or 1 for each word in vocab present in sentence
def bag_of_words(text, vocab, show_details=True):
    # preprocess text
    sentence_words = prepocess_text(text)
    # bag of words - vocabulary matrix - matrix of N words
    bag = [0] * len(vocab)
    for s in sentence_words:
        for i, w in enumerate(vocab):
            if w == s:
                # assign 1 if the word is in vocabulary
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


# predict corresponding intent for input text
def predict_intent(sentence, model):
    p = bag_of_words(sentence, vocabulary, show_details=False)
    # filter out predictions by threshold
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by probability score
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# generate output response based on identified intent of input
def generate_response(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            # select a response at random from available responses
            result = random.choice(i["responses"])
            break
    return result


# generate chatbot output for user input message
def chatbot_response(msg):
    ints = predict_intent(msg, model)
    res = generate_response(ints, intents)
    return res
