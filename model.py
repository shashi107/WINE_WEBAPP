from copyreg import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import VotingClassifier,BaggingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle as plk
from sklearn.metrics import accuracy_score

wine = load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['target']=wine.target

x1=df.drop(['malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','nonflavanoid_phenols','proanthocyanins','target'],axis=1)
y1=df['target']

x_train,x_test,y_train,y_test=train_test_split(x1,y1,train_size=0.8,random_state=2)

bagg=BaggingClassifier(base_estimator= SVC(), n_estimators=36, random_state=2, oob_score=True)
knn=KNeighborsClassifier()
rf=RandomForestClassifier(max_depth=9, max_features='log2', min_samples_leaf=6,min_samples_split=9, n_estimators=36, random_state=2)


models = [('knn', knn), ('rf', rf),('baging',bagg)]
voting = VotingClassifier(estimators=models, voting='hard')
voting.fit(x_train, y_train)

y_pred = voting.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
print('Accuracy :', acc_score)

with open('wine.pkl','wb') as file:
    plk.dump(voting,file)

    