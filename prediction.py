import pandas as pd
# read_file = pd.read_csv (r'BuyAffinity_Test.txt')
# read_file.to_csv (r'BuyAffinity_Test.csv', index=None)
read_file = pd.read_csv (r'BuyAffinity_Train.txt',sep='\t')
read_file.to_csv (r'BuyAffinity_Train.csv', index=None)
df = pd.read_csv('gdrive/My Drive/User Prediction/BuyAffinity_Train.csv')
print(df.head())



import pandas as pd
read_file = pd.read_csv (r'gdrive/My Drive/User Prediction/BuyAffinity_Test.txt',sep='\t')
read_file.to_csv (r'/content/gdrive/User Prediction/BuyAffinity_Test.csv', index=None)
read_file = pd.read_csv (r'BuyAffinity_Train.txt')
read_file.to_csv (r'BuyAffinity_Train.csv', index=None)
data_frame = pd.read_csv('BuyAffinity_Train.csv')

left_rate = df.left.value_counts()/len(df)
left_rate

corr_mat=df.corr()
corr_mat

import matplotlib.pyplot as plt
%matplotlib inline
top_corr_features =corr_mat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
g

df['role']=df['role'].astype('category').cat.codes
df['salary']=df['salary'].astype('category').cat.codes
df.head()

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

Y=df['left']
X=df.drop('left',axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#using logistic Regression
from sklearn import preprocessing
reg = LogisticRegression(solver='lbfgs',max_iter = 15000)
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print ('logistic regression accuracy score:',accuracy_score(y_test,y_predict))


#using DecisionTree
clf=DecisionTreeClassifier(random_state=42)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
print('Decision Tree accuracy score',accuracy_score(y_test,y_predict))



#using Random Forest 
Rtree=RandomForestClassifier()
Rtree.fit(x_train,y_train)
y_predict=Rtree.predict(x_test)
# print(x_test,y_predict)
print('Random forest accuracy score',accuracy_score(y_test,y_predict))



# Random Forest accuracy is high
#Finding important features for leaving the job
important_features=Rtree.feature_importances_
indices=np.argsort(important_features)[::-1]
for i in range(X.shape[1]):
     print ("{}: {} {}".format(i+1,X.columns[indices[i]],importances[indices[i]]))



plt.figure(figsize=(10,6));
plt.bar(range(len(indices)),important_features[indices],color='blue',alpha=0.5,tick_label=X.columns[indices]);
plt.xticks(rotation='-75');



# initiate svm object
clf = svm.SVC()
# Fit the svm object with train data
clf.fit(X_train,y_train)



pred = clf.predict(X_test)

accuracy = accuracy_score(pred, y_test)

print "Accuracy using SVC classifier - ",accuracy


plt.figure(figsize=(24,10));
plt.bar(range(len(indices)),important_features[indices],color='blue',alpha=0.5,tick_label=X.columns[indices]);
plt.xticks(rotation='-75');


import pandas as pd
read_file = pd.read_csv (r'gdrive/My Drive/User Prediction/BuyAffinity_Test.txt',sep='\t')
read_file.to_csv (r'gdrive/My Drive/User Prediction/BuyAffinity_Test.csv', index=None)
read_file = pd.read_csv (r'gdrive/My Drive/User Prediction/BuyAffinity_Train.txt',sep='\t')
read_file.to_csv (r'gdrive/My Drive/User Prediction/BuyAffinity_Train.csv', index=None)




df['F15']=df['F15'].astype('category').cat.codes
df['F16']=df['F16'].astype('category').cat.codes
df.head()

df1.drop(['Index'], axis=1,inplace=True)
df1.insert(0, 'Index', range(1, 1+ len(df1)))
df1['F15']=df1['F15'].astype('category').cat.codes
df1['F16']=df1['F16'].astype('category').cat.codes
