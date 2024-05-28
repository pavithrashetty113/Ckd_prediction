import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

#import Dataset
dataset = pd.read_csv('kidney_disease _final.csv')

from collections import Counter
print(Counter(dataset.classification=='notckd'))

dataset['classification']=dataset['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
dataset.rename(columns={'classification':'class'},inplace=True)

df=dataset
sns.countplot(x='class',data=df)
plt.xlabel("classification")
plt.ylabel("Count")
plt.title("target Class")
plt.show()
#print('Percent of chronic kidney disease sample: ',(round(len(df[df['class']=='ckd'])/len(df['class'])*100,2)),"%");
#print('Percent of not a chronic kidney disease sample: ',(round(len(df[df['class']=='notckd'])/len(df['class'])*100,2)),"%");

dataset.head()

dataset.shape

dataset.dtypes

dataset[['htn','dm','cad','pe','ane']]=dataset[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
dataset[['rbc','pc']] = dataset[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
dataset[['pcc','ba']] = dataset[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
dataset[['appet']] = dataset[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})

# Further cleaning
dataset['pe'] = dataset['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
dataset['appet'] = dataset['appet'].replace(to_replace='no',value=0)
dataset['cad'] = dataset['cad'].replace(to_replace='\tno',value=0)
dataset['dm'] = dataset['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
dataset.drop('id',axis=1,inplace=True)

dataset.head()

for i in ['rc','wc','pcv']:
  dataset[i] = dataset[i].str.extract('(\d+)').astype(float)

dataset=dataset.fillna(dataset.median())

dataset

corr = df.corr()
plt.figure(figsize=(18,10))
sns.set(font_scale=1)
sns.heatmap(corr, annot=True,annot_kws={"size": 8},cmap="YlGnBu")
plt.show()

dataset.drop('pcv',axis=1,inplace=True)
dataset.drop('pot',axis=1,inplace=True)

#Data preprocessing
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


dataset.isnull().sum()

dataset = dataset.dropna(axis=1)

dataset.shape

print(X.iloc[11])
print("#")
print(y.iloc[11])

print(X.iloc[350],y.iloc[350])

print(X.iloc[1],y.iloc[1])

print(X.iloc[399],y.iloc[399])

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
print('#')
print(sc.mean_)
print(sc.var_)
#dump(sc,'scalar_file.joblib')
#sclar=load('scalar_file.joblib')

print(X[11],y[11])

print(X[350],y[350])

print(X[1],y[1])

print(X[399],y[399])

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
cv = KFold(n_splits=20, random_state=1, shuffle=True)

#Splitting the dataset in to training and testing set
X_train , X_test , y_train , y_test   = train_test_split(X,y,test_size = 0.3 , random_state=13)

# import library
from imblearn.over_sampling import SMOTE

smote = SMOTE()

 #fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X_train, y_train)

print('Original dataset shape', Counter(y_train))
print('Resample dataset shape', Counter(y_smote))

#Logistic Regression
# Training the Logistic Regression model on the Training set
lg = LogisticRegression(random_state = 12)
lg.fit(X_train, y_train)

#predicting the test result and also with input
y_pred_lg = lg.predict(X_test)
scores = cross_val_score(lg, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
from numpy import mean
from numpy import std
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
scores

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#calculate accuracy
score_lg = accuracy_score(y_pred_lg,y_test)
score_lg

print("train score - " + str(lg.score(X_train, y_train)))
print("test score - " + str(lg.score(X_test, y_test)))

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lg = confusion_matrix(y_test,y_pred_lg)
sns.set(font_scale=1.4) # for label size
sns.heatmap(cm_lg, annot=True, annot_kws={"size": 16}) # font size

plt.show()

"""ROC curve can efficiently give us the score that how our model is performing in classifing the labels. We can also plot graph between False Positive Rate and True Positive Rate with this ROC(Receiving Operating Characteristic) curve. The area under the ROC curve give is also a metric. Greater the area means better the performance."""

false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred_lg)
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_pred_lg))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(classification_report(y_test, y_pred_lg))

#Support Vector Machine

#fitting SVM to the training set
svm = SVC(kernel='linear', random_state=50)
svm.fit(X_train,y_train)

y_pred_svm = svm.predict(X_test)
scores = cross_val_score(svm, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
from numpy import mean
from numpy import std
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
scores

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#predictin the test result
y_pred_svm = svm.predict(X_test)

score_svm = accuracy_score(y_pred_svm,y_test)
score_svm

print("train score - " + str(svm.score(X_train, y_train)))
print("test score - " + str(svm.score(X_test, y_test)))

#Making the Confusion Matrix
cm_svm = confusion_matrix(y_test,y_pred_svm)

sns.set(font_scale=1.4) # for label size
sns.heatmap(cm_svm, annot=True, annot_kws={"size": 16}) # font size

plt.show()

print(classification_report(y_test, y_pred_svm))

#def ScaleData(ar,means=sc.mean_,stds=sc.var_**0.5):
 # for i in range(0,14):
  #  ar[i]=((ar[i]-means[i])/stds[i])
   # print(a[i])
  #return ar


#K Nearest Neighbors Classifier
#fitting KNN to the training set
knn= KNeighborsClassifier(n_neighbors=3 , metric='minkowski',p=2  )
knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)
scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
from numpy import mean
from numpy import std
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
scores

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#predictin the test result
'''65.000 ,70.000,1.025,0.000,0.000,85.000,20.000,1.000,142.000,4.800,16.100,43.000,9600.000,4.000
'''
#after standardization of 350th row
#[ 0.7972653  -0.48063451  1.41572747 -0.80028958 -0.4377969  -0.84398552
#-0.76031079 -0.3693909   0.48638772  0.06134259  1.31730967  0.50553483
 # 0.47374891 -0.27216573] 0.0


y_pred_knn = knn.predict(X_test)

#calculate accuracy
score_dtc = accuracy_score(y_pred_knn,y_test)
score_dtc

print("train score - " + str(knn.score(X_train, y_train)))
print("test score - " + str(knn.score(X_test, y_test)))

#Making the Confusion Matrix
cm_knn = confusion_matrix(y_test,y_pred_knn)

sns.set(font_scale=1.4) # for label size
sns.heatmap(cm_knn, annot=True, annot_kws={"size": 16}) # font size

plt.show()

print(classification_report(y_test, y_pred_knn))
 
#Random Forest

rf = RandomForestClassifier(random_state=50)
rf.fit(X_train, y_train)

# Predicting the test result
y_pred_rf = rf.predict(X_test)

# Calculating accuracy scores
score_rf = accuracy_score(y_pred_rf, y_test)

print("Train score: ", rf.score(X_train, y_train))
print("Test score: ", rf.score(X_test, y_test))
print("Accuracy score: ", score_rf)

# Making the Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Visualizing the Confusion Matrix
sns.set(font_scale=1.4)  # for label size
sns.heatmap(cm_rf, annot=True, annot_kws={"size": 16})  # font size
plt.show()



pickle.dump(lg,open('lgmodel.pkl','wb'))
kmodel=pickle.load(open('lgmodel.pkl','rb'))

pickle.dump(sc, open('scaler.pkl', 'wb'))
scalr=pickle.load(open('scaler.pkl','rb'))