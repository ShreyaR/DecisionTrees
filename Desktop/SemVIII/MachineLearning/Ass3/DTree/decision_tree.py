"""
Author: Yashoteja Prabhu
Email: yashoteja.prabhu@gmail.com
Date:   25-Mar-2016
"""

"""
This sample code demonstrates how to:
- load text data into python using pandas module
- preprocess textual categorical features
- build decision tree using training data
- predict scores over novel test data
"""

# Loading requisite libraries
import sys
import os
import re
import pandas as pd
import numpy as np
import pdb
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

# This function encodes a symbolic categorical attribute (eg: female/male) as a set of numerical one-versus-all features (one-hot-encoding)
def one_hot_encode_categorical(Xtrn,Xval,Xtst):
    lenc = LabelEncoder()
    catvar = Xtrn.columns[Xtrn.dtypes==object]
    oenc = OneHotEncoder(categorical_features=(Xtrn.dtypes==object),sparse=False)

    #Convert from, say, male/female to 0/1 (refer online for more details)
    for var in catvar:
        lenc.fit( pd.concat( [Xtrn[var],Xval[var],Xtst[var]] ) )
        Xtrn[var] = lenc.transform(Xtrn[var])
        Xval[var] = lenc.transform(Xval[var])
        Xtst[var] = lenc.transform(Xtst[var])

    # one-hot-encoding of 0-(k-1) valued k-categorical attribute
    oenc.fit( pd.concat( [Xtrn,Xval,Xtst] ) )
    Xtrn = pd.DataFrame(oenc.transform(Xtrn))
    Xval = pd.DataFrame(oenc.transform(Xval))
    Xtst = pd.DataFrame(oenc.transform(Xtst))
    return (Xtrn,Xval,Xtst)





# Read training data and partition into features and target label
data = pd.read_csv("train.csv")
Xtrn = data.drop("Survived",1)
Ytrn = data["Survived"]

# Read validation data and partition into features and target label
data = pd.read_csv("validation.csv")
Xval = data.drop("Survived",1)
Yval = data["Survived"]

# Read test data and partition into features and target label
data = pd.read_csv("test.csv")
Xtst = data.drop("Survived",1)
Ytst = data["Survived"]

# convert a symbolic categorical attribute (eg: female/male) to set of numerical one-versus-all features (one-hot-encoding)
Xtrn,Xval,Xtst = one_hot_encode_categorical(Xtrn,Xval,Xtst)


#Dimensionality Reduction of Training, Validation and Test Set
'''pca = PCA(n_components=3)
pca.fit_transform(Xtrn)
pca.transform(Xval)
pca.transform(Xtst)'''

'''
kBest = SelectKBest(chi2, k=5)
kBest.fit(Xtrn,Ytrn)
Xtrn = kBest.transform(Xtrn)
Xtst = kBest.transform(Xtst)
Xval = kBest.transform(Xval)'''




# Build a simple depth-5 decision tree with information gain split criterion
#dtree = DecisionTreeClassifier(criterion="gini", max_depth=11)
#dtree = RandomForestClassifier(n_estimators=5, criterion="entropy", max_depth=4)
#dtree.fit(Xtrn,Ytrn)

#dtree = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=5, class_weight={0:0.65, 1:0.35}, min_samples_split=5)
#dtree = RandomForestClassifier(n_estimators=5, criterion="entropy", max_depth=5, class_weight={0:0.66, 1:0.34}, min_samples_split=5, oob_score=True)
#dtree.fit(Xtrn,Ytrn)



dtree = ExtraTreesClassifier(n_estimators=100, criterion="entropy", max_features=5, max_depth=5, random_state=867, min_samples_split=5)
dtree.fit(Xtrn, Ytrn)


# function score runs prediction on data, and outputs accuracy. 
# If you need predicted labels, use "predict" function
print "\n"

print "training accuracy: ",
print dtree.score(Xtrn,Ytrn)

print "validation accuracy: ",
print dtree.score(Xval,Yval)

print "test accuracy: ",
print dtree.score(Xtst,Ytst)

