import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
#from try_movie import weighted_rating
from try_movie import genre_based
from try_movie import hybrid_based


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except ValueError:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)   
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        #print(type(results))
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #print(results)
        #print(results_index)
        if results[0][results_index]> 0.6:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
        
            print(random.choice(responses))
            #print(tag)
            if tag=="Romance" or tag== "Action" or tag=="Animation" or tag=="Adventure" or tag=="Family" or tag=="Comedy" or tag=="Drama" or tag=="Fantasy" or tag=="Crime" or tag=="Thriller" or tag=="History":
                x=genre_based(tag)
                print(x)
        else:


            print("i Do not understand ,Please type something meaningful")


x="go"
#print("Start talking with the bot(type quit to stop)!")
while(x!="quit"):
    y=int(input("in what way do u want to search movies- 1. explore genre based components and talk with bot, 2.give in user id and get suggestions, 3.quit \n"))
    if y==1:
        chat()
        x="quit"
    elif y==2:
        id1=int(input("Mention your user id \n"))
        title=(input("Metion any one of your favourite movie that you have watched according to you current mood \n"))
        ans=hybrid_based(id1,title)
        print(ans)
        print("        ")
    elif y==3:
        x="quit"
        print("Goodbye ! please cum again \n")
    else:
        print("Please choose the right options \n")
    

