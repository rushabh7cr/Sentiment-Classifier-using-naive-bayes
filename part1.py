import re
import  numpy as np
import sys
import math
from random import seed
from random import randrange
import matplotlib.pyplot as plt
import random
import string
import warnings
def random_generator(size=2, chars=string.ascii_uppercase + string.digits):
            return ''.join(random.choice(chars) for x in range(size))

  ### Format of data in th readme file and detail explanation of the code

def preprocess_data(string):


        pattern = re.sub(r'[?|$|.|!|(|)|;|:|*|,|\|-]', r'', string)       ## Remove special characters
        data=[]

        example = ""
        for line in pattern.split("/n"):

            for word in line.split():
                if word != '0' and word != '1':
                    example = example + " " + word
                else:
                    data.append([example, word])
                    example = ""                                ##Data is stored in list format
        return data


def prepare_vocab(data):
    vocabSet = set([])
    for i in range(0, len(data)):
        for word in data[i][0].split(): vocabSet.add(word)
    return list(vocabSet)




################################################################# Training with Max likelihood
def prob_word_given_class1(training_data,vocabulary):
    posList = []
    positiveCount={}
    for c, v in training_data:
        if v == '1':                                 #First create a list with all words with label 1
            for word in c.split():
                posList.append(word)


    for v in vocabulary:
        counter = 0
        for word in posList:                ## Based on the vocab calc the frequency of each word
            if word==v:
                counter+=1

        positiveCount[v] = np.log(counter/len(posList))                     #calculating the probability i.e P(Word|label)
    return positiveCount

def prob_word_given_class0(training_data,vocabulary):
    negList = []
    for c, v in training_data:
        if v == '0':
            for word in c.split():                          #First create a list with all words with label 0
                negList.append(word)
    negativeCount={}
    for v in vocabulary:
        counter = 0
        for word in negList:                # Based on the vocab calc the frequency of each word
            if v == word:
                counter+=1

        negativeCount[v] = np.log(counter/len(negList))                 #calculating the probability i.e P(Word|label)
    return negativeCount
###############################################################################################

def prob_word_given_class1_map(training_data,vocabulary):
    posList = []
    positiveCount={}
    for c, v in training_data:
        if v == '1':
            for word in c.split():
                posList.append(word)


    for v in vocabulary:
        counter = 0
        for word in posList:
            if word==v:
                counter+=1
        positiveCount[v] = np.log((counter + 1)/(len(posList)+len(vocabulary)))         #calculating the probability i.e P(Word|label) with m=1
    return positiveCount

def prob_word_given_class0_map(training_data,vocabulary):
    negList = []
    for c, v in training_data:
        if v == '0':
            for word in c.split():
                negList.append(word)
    negativeCount={}
    for v in vocabulary:
        counter = 0
        for word in negList:
            if v == word:
                counter+=1
        negativeCount[v] = np.log((counter + 1)/(len(negList)+len(vocabulary)))                 #calculating the probability i.e P(Word|label) with m=1
    return negativeCount
###################################################################################################################
def testing(data,positiveCount,negativeCount,vocab):

    prediction= []
    for example in data:
        prob1 = 0
        prob2 = 0
        for word in example[0].split():
                for k in vocab:                                                             ## example is a sentence
                    if word == k:
                        prob1 += positiveCount[word]                              ## For each word in example add its probability since we are calc Loglikelihood
                        prob2 += negativeCount[word]
                    else:pass
        prior_likelihood1 = prob1 #+ math.log(prior)
        prior_likelihood0 = prob2 #+ math.log(1-prior)

        if prior_likelihood1>prior_likelihood0:
            prediction.append([example[0],'1'])
        elif prior_likelihood1 == prior_likelihood0:
            prediction.append([example[0],random_generator(1, "01")])
        else: prediction.append([example[0],'0'])
    return prediction

def getAccuracy(data, predictions):
    correct = 0
    for i in predictions:
        for k in data:
            if i[0] == k[0]:
                if i[1] == k[1]:
                    correct+=1
                else:pass

    return (correct/(len(predictions)))*100.0

# Split a dataset into k folds
def cross_validation_split(dataset, folds=10):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def perform_validation(folds,data):
    training_data=[]
    test = []
    train=[]
    accuracy2=[]
    accuracy1=[]
    std2=[[],[],[],[],[],[],[],[],[]]
    std1=[[],[],[],[],[],[],[],[],[]]
    error1=[]
    error2=[]
    for i in range(len(folds)):
        test = folds[i]
        #print(test)

        train = folds[:i] + folds[i+1:]
        #print(train)
        sum2 = 0
        sum1 = 0
        training_data=[]
       # print(len(train))
        #print("===")
        for j in range(len(train)):
            for k in train[j]:
                training_data.append(k)

            vocabulary = prepare_vocab(training_data)
            positiveCount_map = prob_word_given_class1_map(training_data, vocabulary)
            negativeCount_map = prob_word_given_class0_map(training_data, vocabulary)
            positiveCount = prob_word_given_class1(training_data,vocabulary)
            negativeCount = prob_word_given_class0(training_data,vocabulary)
            prediction1 = testing(test,positiveCount,negativeCount,vocabulary)
            prediction2 = testing(test,positiveCount_map,negativeCount_map,vocabulary)
            acc1 = getAccuracy(data,prediction1)
            acc2 = getAccuracy(data,prediction2)
            #accuracy1[j] += acc1
            #accuracy2[j] += acc2
            std1[j].append(acc1)
            std2[j].append(acc2)

    for p in std1:

        acc = sum(p)

        accuracy1.append(acc/9)

    for p in std2:

        acc = sum(p)

        accuracy2.append(acc/9)

    for m in std1:
        deviation = np.std(m)

        error1.append(deviation)

    for m in std2:
        deviation = np.std(m)
        error2.append(deviation)


    x = ['100','200','300','400','500','600','700','800','900']
    y = accuracy2
    #plt.subplot(2,1,1)
    plt.errorbar(x,y,yerr = error2,label='M = 1')
    #plt.title("with map")
    plt.ylabel("Accuracy")
    plt.xlabel("Subsamples")

    #plt.subplot(2,1,2)
    plt.errorbar(x, accuracy1, yerr=error1,label = 'M = 0')

    #plt.ylabel("Accuracy")
    #plt.xlabel("Subsamples")
    #function to show the plot
    #plt.legend()
    #plt.show()
    plt.savefig('accuracy_vs_trainingdata.png')



if __name__ == "__main__":
    #seed(1)
    warnings.filterwarnings("ignore")
    string = open(sys.argv[1]).read().lower()
    data = preprocess_data(string)
    #print(data)


    folds = cross_validation_split(data)  ##After partition



    perform_validation(folds,data)