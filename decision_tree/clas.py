import numpy as np
import pandas as pd
import operator
import math


class TTTT:
    def createTree(self, dataSet, labels, depth):
        #print('--------------------')
        # 获得label
        classList = [example[-1] for example in dataSet]
        # 如果标签都是同一类，就保存起来
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 如果特征便利完，依旧分不出来，那就采用多数原则
        if len(dataSet[0]) == 1:
            # 多数表决原则，确定类标签
            return self.mostCnt(classList)
        # 限制树深
        # print('depth:    ', depth)
        if (depth == 0):
            return self.mostCnt(classList)
        depth -= 1
        # 确定出当前最优的分类特征，来进行分类
        bestFeat = self.bestFeature(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        # 删除属性列表中当前分类数据集特征
        subLabels = labels[:]
        del (subLabels[bestFeat])
        # 获取数据集中最优特征所在列，遍历每一个特征取值
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            myTree[bestFeatLabel][value] = self.createTree(self.sliceList(dataSet, bestFeat, value), subLabels,
                                                           depth)
        return myTree

    ###     计算信息熵
    def calcShannon(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 1  ################
            else:
                labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key] / numEntries)
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    ###划分数据集
    def sliceList(self, dataSet, axis, value):
        retdataSet = []
        for featVec in dataSet:
            # 将相同数据特征的提取出来
            if featVec[axis] == value:
                reducedFeatVec = list(featVec[:axis])
                reducedFeatVec.extend(featVec[axis + 1:])
                retdataSet.append(reducedFeatVec)
        return retdataSet

    ###选择最优元素
    def bestFeature(self, dataSet):
        numFeature = len(dataSet[0]) - 1
        baseEntroy = self.calcShannon(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeature):
            featureList = [example[i] for example in dataSet]
            # 去除重复值
            uniqueVals = set(featureList)
            newEntropy = 0.0
            for value in uniqueVals:
                subdataSet = self.sliceList(dataSet, i, value)
                prob = len(subdataSet) / float(len(dataSet))
                newEntropy += prob * np.log2(prob)
            inforGain = baseEntroy - newEntropy
            if inforGain > bestInfoGain:
                bestInfoGain = inforGain
                bestFeature = i
        return bestFeature

    ####选择最多的类标签
    def mostCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        # 排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        # 返回出现次数最多的
        return sortedClassCount[0][0]

    # 预测
    def predict(self, inputTree, featLabels, test):
        classLabel=[]
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if test[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.predict(secondDict[key], featLabels, test)
                else:
                    classLabel = secondDict[key]
        return classLabel
