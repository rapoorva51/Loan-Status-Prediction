#!/usr/bin/env python
# coding: utf-8

# # Loan Status Prediction 

# Importing the Pandas and the Numpy libraries.

# In[2]:


import pandas as pd
import numpy as np


# Importing the dataset 

# In[3]:


df_train = pd.read_csv("train.csv",encoding = 'UTF-8')
df_test = pd.read_csv("test.csv",encoding = 'UTF-8')


# In[4]:


df_train


# In[5]:


df_test


# In[6]:


df_train['which_data'] = 'data_train'
df_test['which_data'] = 'data_test'


# Combining the two datasets into single dataset

# In[7]:


df_all = pd.concat([df_train,df_test],axis = 0)


# In[8]:


df_all


# In[9]:


df_all.info()


# In[10]:


median_value = df_all['ApplicantIncome'].median()
df_all['ApplicantIncome'] = df_all['ApplicantIncome'].fillna(median_value)

median_value = df_all['CoapplicantIncome'].median()
df_all['CoapplicantIncome'] = df_all['CoapplicantIncome'].fillna(median_value)

median_value = df_all['LoanAmount'].median()
df_all['LoanAmount'] = df_all['LoanAmount'].fillna(median_value)

median_value = df_all['Loan_Amount_Term'].median()
df_all['Loan_Amount_Term'] = df_all['Loan_Amount_Term'].fillna(median_value)

median_value = df_all['Credit_History'].median()
df_all['Credit_History'] = df_all['Credit_History'].fillna(median_value)


# In[11]:


df_all.fillna(method = 'ffill',axis = 1)


# Cleaning the dataset

# In[12]:


df_all = df_all.mask(df_all == 0).fillna(df_all.mean())
df_all


# Drop all the null values 

# In[13]:


df_all = df_all.dropna()
print(df_all)


# In[14]:


df_all.isnull().sum()


# In[15]:


df_all.info()


# Multivariate analysis using pairplot

# In[16]:


import seaborn as sns
sns.pairplot(df_all)


# In[17]:


df1 = pd.get_dummies(df_all,columns = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status','which_data'],drop_first = True)


# In[18]:


df_all.shape


# Checking and Assigning the unique values of the columns

# In[19]:


df_all['Gender'].unique()
df_all['Gender'].value_counts()


# In[20]:


df_all['Married'].unique()
df_all['Married'].value_counts()


# In[21]:


df_all['Self_Employed'].unique()
df_all['Self_Employed'].value_counts()


# In[22]:


df_all['Education'].unique()
df_all['Education'].value_counts()


# In[23]:


df_all['Property_Area'].unique()
df_all['Property_Area'].value_counts()


# In[24]:


df_all['Loan_Status'].unique()
df_all['Loan_Status'].value_counts()


# In[25]:


Value_Mapping = {'Yes' : 1, 'No' : 0}
df_all['Married_Section'] = df_all['Married'].map(Value_Mapping)
df_all.head(5)


# In[26]:


Value_Mapping1 = {'Male' : 1, 'Female' : 0}
df_all['Gender_Section'] = df_all['Gender'].map(Value_Mapping1)
df_all.head(5)


# In[27]:


Value_Mapping2 = {'Graduate' : 1, 'Not Graduate' : 0}
df_all['Edu_Section'] = df_all['Education'].map(Value_Mapping2)
df_all.head(5)


# In[28]:


Value_Mapping3 = {'Yes' : 1, 'No' : 0}
df_all['Employed_Section'] = df_all['Self_Employed'].map(Value_Mapping3)
df_all.head(5)


# In[29]:


Value_Mapping4 = {'Y' : 1, 'N' : 0}
df_all['Loan_Section'] = df_all['Loan_Status'].map(Value_Mapping4)
df_all.head(5)


# In[30]:


Value_Mapping5 = {'Urban' : 1, 'Rural' : 0,'Semiurban':2}
df_all['Propertyarea_Section'] = df_all['Property_Area'].map(Value_Mapping5)
df_all.head(5)


# In[31]:


Value_Mapping = {'Y' : 1, 'N' : 0}
df_all['Loan_Status'] = df_all['Loan_Status'].map(Value_Mapping)
df_all.head(5)


# In[32]:


df = df_all.drop(['Loan_ID','Gender','Married','Education','Self_Employed','Property_Area','which_data'],axis = 1)
df


# Finding the correlation 

# In[33]:


correlation = df.corr()
sns.heatmap(correlation,xticklabels = correlation.columns,yticklabels = correlation.columns,annot = True)


# In[34]:


df


# In[35]:


df=df.select_dtypes(include=['float64','int64'])


# In[36]:


y = df["Loan_Status"]
X = df.drop("Loan_Status",axis=1)


# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42)


# In[38]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[39]:


from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()


# In[40]:


forest.fit(X_train,y_train)


# In[41]:


forest.score(X_test,y_test)


# In[42]:


from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor()  
regressor.fit(X, y)


# In[43]:


regressor.score(X_test,y_test)


# In[44]:


from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor.fit(X, y)  
regressor.score(X_test,y_test)


# In[45]:


from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
estimator = [] 
estimator.append(('LR',  
                  LogisticRegression(solver ='lbfgs',  
                                     multi_class ='multinomial',  
                                     max_iter = 200))) 
estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
estimator.append(('DTC', DecisionTreeClassifier())) 
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
vot_hard.fit(X_train, y_train) 
y_pred = vot_hard.predict(X_test) 
score = accuracy_score(y_test, y_pred) 
print("Hard Voting Score % d" % score) 
vot_soft = VotingClassifier(estimators = estimator, voting ='soft') 
vot_soft.fit(X_train, y_train) 
y_pred = vot_soft.predict(X_test) 
score = accuracy_score(y_test, y_pred) 
print("Soft Voting Score % d" % score) 


# In[48]:


import xgboost as xgb 
my_model = xgb.XGBClassifier() 
my_model.fit(X_train, y_train) 


# In[49]:


y_pred = my_model.predict(X_test) 


# In[50]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
cm


# In[51]:


from sklearn.metrics import mean_squared_error as MSE 
import xgboost as xg 
xgb_r = xg.XGBRegressor(objective ='reg:linear', 
                  n_estimators = 10, seed = 123) 
xgb_r.fit(X_train, y_train) 
pred = xgb_r.predict(X_test) 
rmse = np.sqrt(MSE(y_test, pred)) 
print("RMSE : % f" %(rmse)) 


# In[52]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)  


# In[53]:


from sklearn.svm import SVC 
clf = SVC(kernel='linear') 
clf.fit(X_train, y_train) 
clf.score(X_test,y_test)

