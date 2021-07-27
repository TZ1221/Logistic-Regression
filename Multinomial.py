from __future__ import division
import math
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Multinomial:
    def __init__(self):
        self.priorProbability = {}
        self.conditionalProbability = {}

    def GetWordFrequencies(self, Data):
        freq_dic = {}
        for line in open(Data).readlines():
            documentId, wordId, count = [int(s) for s in line.split()]
            if wordId not in freq_dic:
                freq_dic[wordId] = count
            else:
                freq_dic[wordId] += count   
        return freq_dic


    def MultinomialTrain(self, Data, Label, vocabulary):
        self.priorProbability = {}
        self.conditionalProbability = {}

        document = [int(s) for s in open(Label).read().split()]
        documentSize = len(document)
        frequency = dict(Counter(document))

        self.priorProbability = dict(map(lambda x: (x[0], math.log(x[1] / documentSize)), frequency.items()))

        frequency_dictionary = {}
        classWordFrequency = {}
        for eachline in open(Data).readlines():
            documentId, wordId, count = [int(s) for s in eachline.split()]
            if wordId in vocabulary:
                documentClass = document[documentId - 1]

                if documentClass in classWordFrequency:
                    classWordFrequency[documentClass] += count
                else:
                    classWordFrequency[documentClass] = count
                   

                if documentClass not in frequency_dictionary:
                    frequency_dictionary[documentClass] = {}
                    frequency_dictionary[documentClass][wordId] = {}
                    frequency_dictionary[documentClass][wordId][documentId] = count
                else:
                    if wordId not in frequency_dictionary[documentClass]:
                        frequency_dictionary[documentClass][wordId] = {}
                        frequency_dictionary[documentClass][wordId][
                            documentId] = count
                    elif documentId not in frequency_dictionary[documentClass][
                            wordId]:
                        frequency_dictionary[documentClass][wordId][
                            documentId] = count
                    else:
                        frequency_dictionary[documentClass][wordId][
                            documentId] += count

        wordsSize = len(vocabulary)
        for documentClass in self.priorProbability.keys():
            self.conditionalProbability[documentClass] = {}
            for wordId in vocabulary:
                Sum=(sum(frequency_dictionary.get(documentClass, {}).get(wordId, {}).values()) + 1) 
                Size=(classWordFrequency[documentClass] + wordsSize)
                self.conditionalProbability[documentClass][wordId] = math.log(Sum/Size)



    def test(self, testData, vocabulary):
        documentProbabilities = {}
        for line in open(testData).readlines():
            documentId, wordId, count = [int(s) for s in line.split()]
            if documentId not in documentProbabilities:
                documentProbabilities[documentId] = {}
            for documentClass in self.priorProbability.keys():
                if documentClass not in documentProbabilities[documentId]:
                    documentProbabilities[documentId][
                        documentClass] = self.priorProbability[documentClass]
                if wordId in vocabulary:
                    documentProbabilities[documentId][
                        documentClass] += count * self.conditionalProbability[
                            documentClass][wordId]
        result=dict(map(lambda x: (x[0], max(x[1], key=x[1].get)),documentProbabilities.items()))
        return result

    def calculateMetrics(self, testLabel, predictions):
        testClasses = [int(s) for s in open(testLabel).read().split()]
        predictedClasses = []
        for documentId in range(1, len(testClasses) + 1):
            predictedClasses.append(predictions[documentId])

        accuracy = accuracy_score(testClasses, predictedClasses)
        precision = precision_score(testClasses, predictedClasses, average="weighted")
        recall = recall_score(testClasses, predictedClasses, average="weighted")

        return accuracy, precision, recall

    def calculateClassMetrics(self, testLabel, predictions):
        testClasses = [int(s) for s in open(testLabel).read().split()]
        predictedClasses = []
        for documentId in range(1, len(testClasses) + 1):
            predictedClasses.append(predictions[documentId])

        precision = precision_score(testClasses, predictedClasses, average=None)
        recall = recall_score(testClasses, predictedClasses, average=None)

        print("printing precision value::\n{}".format(precision))
        print("printing Recall value::\n{}".format(recall))

    def run(self, trainData, trainLabel, testData, testLabel):
        frequency_dictionary = self.GetWordFrequencies(trainData)
        sortedWords = sorted(frequency_dictionary, key=frequency_dictionary.get, reverse=True)

        vocabularySize = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000,len(sortedWords)]
        #vocabularySize = [50]

        print("Size\tAccuracy\tPrecision\tRecall")
        for size in vocabularySize:
            vocabulary = sortedWords[:size]
            self.MultinomialTrain(trainData, trainLabel, vocabulary)
            predictions = self.test(testData, vocabulary)

            accuracy, precision, recall = self.calculateMetrics(
                testLabel, predictions)
            print("{}\t{}\t{}\t{}".format(size, accuracy, precision, recall))

            if size == len(sortedWords):
                self.calculateClassMetrics(testLabel, predictions)
