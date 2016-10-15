from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier


#data and labels
X = [[181,234,234], [23,546,575], [325,455,654], [253,345,642],[535,786,765],[435,465,726],[232,454,676],[232,151,181]]
Y = ['male', 'female', 'female','male','male','female','male','female']


#classifiers
#using the default value of all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_KNN = KNeighborsClassifier()
clf_RC = RidgeClassifier()

#training models

clf_tree.fit(X,Y)
clf_RC.fit(X,Y)
clf_KNN.fit(X,Y)

#testing using the new data

predict_tree = clf_tree.predict([[223,435,121]])
predict_KNN = clf_KNN.predict([[223,435,121]])
predict_RC =clf_RC.predict([[223,435,121]])

print predict_RC
print predict_KNN
print predict_tree


