from __future__ import division
from sklearn.model_selection import KFold

import numpy as np
import random
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self,dataSet,learningRate=0.001,tolerance=0.001):
        self.dataSet = dataSet.fillna(0)
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.maxDepth = 1000
        self.kFold = 10
        self.kf = KFold(n_splits=self.kFold, shuffle=True)

    def normalize(self, dataSet,means=[],stds=[]):
        normalized = dataSet.copy(True)
        attributes = dataSet.shape[1] - 1
        if len(means) != attributes:
            newMeans = []
            newStds = []
            for i in range(attributes):
                mean = dataSet[i].mean()
                newMeans.append(mean)
                std = dataSet[i].std()
                newStds.append(std)
                normalized[i] = normalized[i].apply(lambda x: (x - mean) / std if std > 0 else 0)
            return normalized, newMeans, newStds

        elif len(means) == attributes:
            for i in range(attributes):
                mean = means[i]
                std = stds[i]
                normalized[i] = normalized[i].apply(lambda x: (x - mean) / std if std > 0 else 0)
            return normalized, means, stds
        
        
    def constantFeature(self, dataSet):
        result = dataSet.copy(True)
        result.columns = range(1, result.shape[1] + 1)
        result.insert(0, 0, 1)
        return result

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def hypothesisProbability(self, x, theta):
        result= 1 / (1 + np.exp(-np.dot(x, theta)))
        return result

    def CalculateCost(self, x, y, theta):
        probability = self.hypothesisProbability(x, theta)
        result=(np.multiply(-y, np.log(probability)) - np.multiply((1 - y), np.log(1 - probability))).mean()
        return result

    def gradient(self, X, Y, theta):
        prediction = self.hypothesisProbability(X, theta)
        error = prediction - Y

        return np.dot(X.T, error) / X.shape[0]

    def logisticRegression(self, dataSet, plotGraph):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        self.attributes = dataSet.shape[1] - 1
        theta = np.matrix(np.zeros(self.attributes)).T

        logisticLoss = [self.CalculateCost(X, Y, theta)]
        for i in range(self.maxDepth):
            gradient = self.gradient(X, Y, theta)
            newTheta = theta - self.learningRate * gradient
            newCost = self.CalculateCost(X, Y, newTheta)
            if (logisticLoss[i] - newCost < self.tolerance):
                break
            else:
                logisticLoss.append(newCost)
                theta = newTheta

        if plotGraph:
            plt.plot(logisticLoss)
            plt.xlabel('Iteration')
            plt.ylabel('Logistic Loss')
            plt.title('Logistic Regression')
            plt.show()

        return theta

    def predictResult(self, X, theta):
        probabilities = self.hypothesisProbability(X, theta)

        return [
            1 if probability >= 0.5 else 0 for probability in probabilities
        ]

    def GetAccuracy(self, dataSet, theta):
        Y = dataSet.iloc[:, -1]
        X = np.matrix(dataSet.iloc[:, :-1])
        
        prediction = self.predictResult(X, theta)
        GetAccuracy = (prediction == Y).mean()

        Y = Y.values.tolist()
        truePositive = 0
        for index, value in enumerate(prediction):
            if value == 1 and Y[index] == 1:
                truePositive += 1

        precision = truePositive / prediction.count(1)
        recall = truePositive / Y.count(1)

        return GetAccuracy, precision, recall

    def validate(self):
        trainAccuracies = []
        trainPrecisions = []
        trainRecalls = []
        
        testAccuracies = []
        testPrecisions = []
        testRecalls = []

        fold = 1
        plotFold = random.randint(1, self.kFold + 1)

        print(
            "Fold\tTrained Accuracy\tTrained Precision\tTrained Recall\tTested Accuracy\tTested Precision\tTested Recall"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet, trainAttributeMeans, trainAttributeStds = self.normalize(self.dataSet.iloc[trainIndex])
            trainDataSet = self.constantFeature(trainDataSet)
            theta = self.logisticRegression(trainDataSet, fold == plotFold)
            trainAccuracy, trainPrecision, trainRecall = self.GetAccuracy(trainDataSet, theta)
            trainAccuracies.append(trainAccuracy)
            trainPrecisions.append(trainPrecision)
            trainRecalls.append(trainRecall)

            testDataSet, _, _ = self.normalize(self.dataSet.iloc[testIndex], trainAttributeMeans,trainAttributeStds)
            testDataSet = self.constantFeature(testDataSet)
            testAccuracy, testPrecision, testRecall = self.GetAccuracy(testDataSet, theta)
            testAccuracies.append(testAccuracy)
            testPrecisions.append(testPrecision)
            testRecalls.append(testRecall)

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, trainAccuracy, trainPrecision, trainRecall, testAccuracy,testPrecision, testRecall))
            fold += 1

        print("Mean value\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.mean(trainAccuracies), np.mean(trainPrecisions),
            np.mean(trainRecalls), np.mean(testAccuracies),
            np.mean(testPrecisions), np.mean(testRecalls)))
        print("Standard Deviation value\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.std(trainAccuracies), np.std(trainPrecisions),
            np.std(trainRecalls), np.std(testAccuracies),
            np.std(testPrecisions), np.std(testRecalls)))
