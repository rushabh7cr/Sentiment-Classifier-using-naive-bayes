# Sentiment-Classifier-using-naive-bayes

For each of the 3 datasets run stratified cross-validation to generate learning curves for Naive Bayes with m = 0 and with m = 1. For each dataset, plot averages of the accuracy and standard deviations (as error bars) as a function of the train set size.

In this we run stratified cross validation to generate learning curves for naive bayes with m=0 and m=1.

SKELETON OF CODE:

1st PART: Import DATA
```
string = open(sys.argv[1]).read().lower()
```

2nd PART: Clean data
Here we remove special characters and store the data in a list format [[example,class]].
```
Data = [' a very very very slowmoving aimless movie about a distressed drifting young man', '0'], [' not sure who was more lost the flat characters or the audience nearly half of whom walked out', '0']
```

3rd PART: Cross validation
We create 10 random folds each fold consisting of 100 examples and their class resp.
```
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
```
4th PART: Here we do the subSampling.
We have already split our data into 10 folds. We assign 1 fold as test set and the other 9 as training.
Now, in the second loop of the code below, we subsample data.
And for each subsample data calc the vocabulary,probability and accuracy.
```
for i in range(len(folds)):
    test = folds[i]
    #print(test)

    train = folds[:i] + folds[i+1:]
    #print(train)
    sum = 0
    training_data=[]
    for j in train:
        for k in j:
            training_data.append(k)
        vocabulary = prepare_vocab(training_data)

    for k in range(len(m)):

        positiveCount_map = prob_word_given_class1_map(training_data, vocabulary,m[k])
        negativeCount_map = prob_word_given_class0_map(training_data, vocabulary,m[k])
        prediction2 = testing(test,positiveCount_map,negativeCount_map,vocabulary)
        acc = getAccuracy(data,prediction2)
        accuracy[k] +=acc
        std[k].append(acc)
```

5th PART:Now we perform validation
That is we split the data in training and Test set. Using the training set we find out the vocabulary( i.e unique words) and the calculate P(word|class) and store it in a dictionary.
The dictionary contains key as word and value as its log(probability).
```
def prepare_vocab(data):
    vocabSet = set([])
    for i in range(0, len(data)):
        for word in data[i][0].split(): vocabSet.add(word)
    return list(vocabSet)

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
        positiveCount[v] = np.log((counter + 1)/(len(posList)+len(vocabulary)))
    return positiveCount
```
We call these functions in another functoin called 
def cross_validation_split()


6th PART: We calculate the likelihood of each example of our test data corresponding to the probabilities given by training data.
LogLikelihood = log(word1|class) + log(word2|class)..... 
```
def testing(data,positiveCount,negativeCount,vocab):

    prediction= []
    for example in data:
        prob1 = 0
        prob2 = 0
        for word in example[0].split():
                for k in vocab:
                    if word == k:
                        prob1 += positiveCount[word]
                        prob2 += negativeCount[word]
                    else:pass
        prior_likelihood1 = prob1         
	 prior_likelihood0 = prob2

        if prior_likelihood1>prior_likelihood0:
            prediction.append([example[0],'1'])
        elif prior_likelihood1 == prior_likelihood0:
            prediction.append([example[0],random_generator(1, "01")])
        else: prediction.append([example[0],'0'])
    return prediction
```

7th PART: Calculate the accuracy of the given prediction
```
def getAccuracy(data, predictions):
    correct = 0
    for i in predictions:
        for k in data:
            if i[0] == k[0]:
                if i[1] == k[1]:
                    correct+=1
                else:pass

    return (correct/(len(predictions)))*100.0
```

8th PART: PLOT
```
plt.errorbar(x,y,yerr = error2,label='M = 1')
```

