#!/usr/bin/env python
# -*-coding:utf-8-*-
# 决策树的建立，训练测试，
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# 读数据
allElectronicsData = open(r'train.csv', 'r')
reader = csv.reader(allElectronicsData)
headers = next(reader)
print(headers)

featureList = []
labelList = []
# 分析数据
for row in reader:
    # print(row)
    if (row):
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

print(featureList)
# 转化数据
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print('dummyX:' + str(dummyX))
print(vec.get_feature_names())

print('labelList:' + str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print('dummyY:' + str(dummyX))

# 训练数据
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print('clf' + str(clf))
# 转化为dot模式
with open('iris.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# 决策树的预测
oneRowX = dummyX[0, :]
print('oneRowX:' + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print('newRowX:' + str(newRowX))

predictedY = clf.predict(newRowX)
print('predictedY:' + str(predictedY))