from __future__ import division
from collections import Counter
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score


class MultivariateBernoulli:
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

    def MultivariateBernoullitrain(self, trainData, trainLabel, voca):
        self.priorProbability = {}
        self.conditionalProbability = {}

        documentClasses = [int(s) for s in open(trainLabel).read().split()]
        totalDocuments = len(documentClasses)
        classFrequency = dict(Counter(documentClasses))

        self.priorProbability = dict(map(lambda x: (x[0], math.log(x[1] / totalDocuments)),classFrequency.items()))

        frequency_dictionary = {}
        for line in open(trainData).readlines():
            document, word, count = [int(s) for s in line.split()]
            if word in voca:
                documentClass = documentClasses[document - 1]
                if documentClass not in frequency_dictionary:
                    frequency_dictionary[documentClass] = {}
                    frequency_dictionary[documentClass][word] = {}
                    frequency_dictionary[documentClass][word][document] = count
                else:
                    if word not in frequency_dictionary[documentClass]:
                        frequency_dictionary[documentClass][word] = {}
                        frequency_dictionary[documentClass][word][document] = count
                    elif document not in frequency_dictionary[documentClass][word]:
                        frequency_dictionary[documentClass][word][document] = count
                    else:
                        frequency_dictionary[documentClass][word][document] += count

        totalClasses = len(self.priorProbability.keys())
        for documentClass in self.priorProbability.keys():
            self.conditionalProbability[documentClass] = {}
            for word in voca:
                self.conditionalProbability[documentClass][word] = math.log(
                    (len(frequency_dictionary.get(documentClass, {}).get(word, {}).keys()) + 1) /
                    (classFrequency[documentClass] + totalClasses))

    def test(self, testData, voca):
        documentProbabilities = {}
        for line in open(testData).readlines():
            document, word, count = [int(s) for s in line.split()]
            if document not in documentProbabilities:
                documentProbabilities[document] = {}
            for documentClass in self.priorProbability.keys():
                if documentClass not in documentProbabilities[document]:
                    documentProbabilities[document][
                        documentClass] = self.priorProbability[documentClass]
                if word in voca:
                    documentProbabilities[document][
                        documentClass] += count * self.conditionalProbability[
                            documentClass][word]
        result=dict(map(lambda x: (x[0], max(x[1], key=x[1].get)),documentProbabilities.items()))
        return result

    def calculateMetrics(self, testLabel, predictions):
        testClasses = [int(s) for s in open(testLabel).read().split()]
        predictedClasses = []
        for document in range(1, len(testClasses) + 1):
            predictedClasses.append(predictions[document])

        accuracy = accuracy_score(testClasses, predictedClasses)
        precision = precision_score(
            testClasses, predictedClasses, average="weighted")
        recall = recall_score(
            testClasses, predictedClasses, average="weighted")

        return accuracy, precision, recall

    def calculateClassMetrics(self, testLabel, predictions):
        testClasses = [int(s) for s in open(testLabel).read().split()]
        predictedClasses = []
        for document in range(1, len(testClasses) + 1):
            predictedClasses.append(predictions[document])

        precision = precision_score(
            testClasses, predictedClasses, average=None)
        recall = recall_score(testClasses, predictedClasses, average=None)

        print("precision value::\n{}".format(precision))
        print("Recall value::\n{}".format(recall))

    def run(self, trainData, trainLabel, testData, testLabel):
        frequency_dictionary = self.GetWordFrequencies(trainData)
        sortedWords = sorted( frequency_dictionary, key=frequency_dictionary.get, reverse=True)

        vocabularySize = [
            100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000,
            len(sortedWords)
        ]

        print("Size\tAccuracy\tPrecision\tRecall")
        for size in vocabularySize:
            vocabulary = sortedWords[:size]
            self.MultivariateBernoullitrain(trainData, trainLabel, vocabulary)
            predictions = self.test(testData, vocabulary)

            accuracy, precision, recall = self.calculateMetrics(testLabel, predictions)
            print("{}\t{}\t{}\t{}".format(size, accuracy, precision, recall))

            if size == len(sortedWords):
                self.calculateClassMetrics(testLabel, predictions)
