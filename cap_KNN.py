# cap_KNN.py

import pandas as pd
import sklearn.cross_validation as skcv
from sklearn import neighbors
from sklearn.ensemble import BaggingClassifier
import math

df = pd.read_csv('waveform.data', header=None)
colnames = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','class']
df.columns = colnames
# colnames from feature extraction
best10 = ['f11', 'f7', 'f15', 'f6', 'f12', 'f10', 'f13', 'f5', 'f9', 'f16']
# the reduced data set
reduced = df[best10]
# split
X = reduced
y = df['class']
xtrain, xtest, ytrain, ytest = skcv.train_test_split(X, y, test_size=0.33, random_state=0)
# state the split sizes
print "Xtrain: ", len(xtrain) # 3350
print "Xtest: ", len(xtest) # 1650
# define k to be the square root of the number of train instances
k = int(math.sqrt(len(xtrain))) # per Duda et al
print "Using k = ", k  # 57

# KNN with 57 neighbors and 10 features
print "Running KNN w/ 57 neighbors and the 10 features found earlier: "
neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
ypred = neigh.fit(xtrain, ytrain).predict(xtest)
diff = ypred - ytest
count = 0 
for i in range(len(diff)):
	if diff[i] == 0:
		count = count + 1

acc1 = float(count)/float(len(diff)) * 100
print "Accuracy with 10 features: " + str(round(acc1, 1)) + "%"
# 85.3%, but probably down to correlation w/ class data

# KNN with 57 neighbors and all (21) features
print "Running KNN w/ 57 neighbors and all features, for comparison: "
X = df[colnames[0:21]]
y = df['class']
xtrain, xtest, ytrain, ytest = skcv.train_test_split(X, y, test_size=0.33, random_state=0)
neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
ypred = neigh.fit(xtrain, ytrain).predict(xtest)
diff = ypred - ytest
count = 0
for i in range(len(diff)):
	if diff[i] == 0:
		count = count + 1

acc2 = float(count)/float(len(diff)) * 100
print "Accuracy with all features: " + str(round(acc2, 1)) + "%"
# 86.0%
print "Improvement using all features, vs top 10: " + str(round((acc2 - acc1)/acc1 * 100, 1)) + "%"
# 0.9% 

# KNN with 57 neighbors and only 6 features
print "Running KNN w/ 57 neighbors and the top 6 features: "
best6 = ['f11', 'f7', 'f15', 'f6', 'f12', 'f10']
reduced = df[best6]
X = reduced
y = df['class']
xtrain, xtest, ytrain, ytest = skcv.train_test_split(X, y, test_size=0.33, random_state=0)
neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
ypred = neigh.fit(xtrain, ytrain).predict(xtest)
diff = ypred - ytest
count = 0
for i in range(len(diff)):
	if diff[i] == 0:
		count = count + 1

acc3 = float(count)/float(len(diff)) * 100
print "Accuracy with top 6 features: " + str(round(acc3, 1)) + "%"
# 80.3% - still likely down to correlation
print "Improvement using all features, vs top 6: " + str(round((acc2 - acc3)/acc3 * 100, 1)) + "%"
# 7.1%

# bagging experiment
print "Running bagging meta-estimator w/ KNN w/ 57 neighbors and 10 max features: "
bagging = BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors=k), max_samples=0.5, max_features=10)
X = df[colnames[0:21]]
y = df['class']
xtrain, xtest, ytrain, ytest = skcv.train_test_split(X, y, test_size=0.33, random_state=0)
ypred = bagging.fit(xtrain, ytrain).predict(xtest)
diff = ypred - ytest
count = 0

for i in range(len(diff)):
	if diff[i] == 0:
		count = count + 1

acc4 = float(count)/float(len(diff)) * 100
print "Accuracy with bagging and 10 features max: " + str(round(acc4, 1)) + "%"
# 85.2 % - quite good, but which features were chosen?

# Run with the columns most poorly correlated with the class column
print "Running KNN w/ 57 neighbors and the features most poorly correlated with the class data: "
worst = ['f1', 'f9', 'f17', 'f18', 'f19', 'f20', 'f21']
X = df[worst]
y = df['class']

neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
ypred = neigh.fit(xtrain, ytrain).predict(xtest)
diff = ypred - ytest
count = 0
for i in range(len(diff)):
	if diff[i] == 0:
		count = count + 1

acc5 = float(count)/float(len(diff)) * 100
print "Accuracy with 7 least correlated features: " + str(round(acc5, 1)) + "%"
# 86.0%
print "Difference with results using all features: " + str(round((acc2 - acc5)/acc5 * 100, 1)) + "%"
# 0%