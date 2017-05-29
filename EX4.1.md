
# First go- Linear SVM
## Data preparation

First thing we want to do is import all of the relevant file and use the cross validation function to test our model before the test file.

%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.neighbors
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm

cd C:\Users\Alon\Desktop\Alon\School\4th year\SemesterB\Data science\ex4\


```python
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

```

Now we can load the train file and address all the issues:
1. Fill all of the numeric features null values with the mean value.
2. The rest of the null values we weill feel using the majority rule.
3. Create some new features- Total income and loan amount, both of them we will normalize using the LOG function.


```python
df = pd.read_csv('data/train.csv')
```


```python
df['Gender'] = df['Gender'].fillna( df['Gender'].dropna().mode().values[0] )
df['Married'] = df['Married'].fillna( df['Married'].dropna().mode().values[0] )
df['Dependents'] = df['Dependents'].fillna( df['Dependents'].dropna().mode().values[0] )
df['Self_Employed'] = df['Self_Employed'].fillna( df['Self_Employed'].dropna().mode().values[0] )
df['LoanAmount'] = df['LoanAmount'].fillna( df['LoanAmount'].dropna().mean() )
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna( df['Loan_Amount_Term'].dropna().mode().values[0] )
df['Credit_History'] = df['Credit_History'].fillna( df['Credit_History'].dropna().mode().values[0] )
df['Dependents'] = df['Dependents'].str.rstrip('+')
df['Gender'] = df['Gender'].map({'Female':0,'Male':1}).astype(np.int)
df['Married'] = df['Married'].map({'No':0, 'Yes':1}).astype(np.int)
df['Education'] = df['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
df['Self_Employed'] = df['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)
df['Loan_Status'] = df['Loan_Status'].map({'N':0, 'Y':1}).astype(np.int)
df['Dependents'] = df['Dependents'].astype(np.int)
df['LoanAmount'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
TotalIncome = df['TotalIncome']
df['TotalIncome'] = np.log(df['TotalIncome'])
df.drop(labels=['TotalIncome'], axis=1,inplace = True)
df.insert(8, 'TotalIncome', TotalIncome)
```

## Set the model
* Now we want to start working on the model, first we change all of the features to str types in order to make our lifes a bit easier.
* After that we'll use the pandas commands to get the relvent features for the model
* next we'll use the fit transform function to turn our df into a matrix.


```python
from sklearn.preprocessing import LabelEncoder
var_mod = list(df.columns.values)
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i].astype(str))
```


```python
X,y  = df.iloc[:, 1:-1], df.iloc[:, -1]
X= pd.get_dummies(X)
```


```python
from sklearn.preprocessing import StandardScaler
slc= StandardScaler()
X_train_std = slc.fit_transform(X)
```

## SVM Model
we'll run the model on some relevent features, we noticed that in wasn't criticle and as long as we ran the Linear SVM, the results stayed the same.


```python
from sklearn import svm
clf = svm.SVC(kernel='linear')
#var_mod.remove('Loan_Status')
var1=['TotalIncome','LoanAmount','Credit_History','Dependents','Married','Education']
classification_model(clf, df,var1,'Loan_Status')
clf.fit(X_train_std, y)
```

    Accuracy : 80.945%
    Cross-Validation Score : 80.946%
    




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



## Test File:
Now let's load the test file and repeat the proccess that we did for the train file


```python
dtest = pd.read_csv('data/test.csv')
dtest['Gender'] = dtest['Gender'].fillna( dtest['Gender'].dropna().mode().values[0])
dtest['Married'] = dtest['Married'].fillna( dtest['Married'].dropna().mode().values[0])
dtest['Dependents'] = dtest['Dependents'].str.rstrip('+')
dtest['Dependents'] = dtest['Dependents'].fillna( dtest['Dependents'].dropna().mode().values[0]).astype(np.int)
dtest['Self_Employed'] = dtest['Self_Employed'].fillna( dtest['Self_Employed'].dropna().mode().values[0])
dtest['LoanAmount'] = dtest['LoanAmount'].fillna( dtest['LoanAmount'].dropna().mode().values[0])
dtest['Loan_Amount_Term'] = dtest['Loan_Amount_Term'].fillna( dtest['Loan_Amount_Term'].dropna().mode().values[0])
dtest['Credit_History'] = dtest['Credit_History'].fillna( dtest['Credit_History'].dropna().mode().values[0] )
dtest['Gender'] = dtest['Gender'].map({'Female':0,'Male':1})
dtest['Married'] = dtest['Married'].map({'No':0, 'Yes':1}).astype(np.int)
dtest['Education'] = dtest['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
dtest['Self_Employed'] = dtest['Self_Employed'].map({'No':0, 'Yes':1})
dtest['LoanAmount'] = np.log(dtest['LoanAmount'])
dtest['TotalIncome'] = dtest['ApplicantIncome'] + dtest['CoapplicantIncome']
TotalIncome = dtest['TotalIncome']
dtest['TotalIncome'] = np.log(dtest['TotalIncome'])
dtest.drop(labels=['TotalIncome'], axis=1,inplace = True)
dtest.insert(8, 'TotalIncome', TotalIncome)
```


```python
from sklearn.preprocessing import LabelEncoder
var_mod = list(dtest.columns.values)
var_mod.remove('Loan_ID')
le = LabelEncoder()
for i in var_mod:
   dtest[i] = le.fit_transform(dtest[i].astype(str))
```

### Now we can use the predict function and create our submission file.


```python
X_test = dtest.iloc[:,1:]
X_test= pd.get_dummies(X_test)
X_test_std = slc.transform(X_test)
y_test_pred = clf.predict(X_test_std)
dtest['Loan_Status'] = y_test_pred
```


```python
df_final = dtest.drop(['Gender', 'Married', 'Dependents','TotalIncome', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], axis=1)
df_final['Loan_Status'] = df_final['Loan_Status'].map({0:'N', 1:'Y'})
df_final.to_csv('my_submission1.csv', index=False)
```

## Results:

We got a grade of 0.77778 and we're at place number 790(alonma)

<img src="https://github.com/alonma/HW4/blob/master/Capture1.JPG" />
