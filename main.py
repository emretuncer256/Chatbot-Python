import pickle
import nltk
import json
import random
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

with open("training_data", "rb") as f:
    data = pickle.load(f)

model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, word in enumerate(words):
            if s == word:
                bag[i] = 1
                if show_details:
                    print("Found in bag %s" % word)
    return np.array(bag)


def classify(sentence):
    # generate probabilities from the model
    ERROR_THRESHOLD = 0.25
    results = model.predict(np.array([bag_of_words(sentence, data["words"])]), verbose=0)[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": data["classes"][r[0]], "probability": str(r[1])})
    return return_list


with open('intents.json') as json_data:
    intents = json.load(json_data)


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0]['intent']:
                    return random.choice(i['responses'])

            results.pop(0)


if __name__ == "__main__":
# print(response("what is your name?")) # ex. output: I'm just a chat agent. I only exist in the internet

    while True:
        print("You: ", end="")
        user_input = input()
        if user_input == "quit":
            break
        print("Bot: " + response(user_input))
