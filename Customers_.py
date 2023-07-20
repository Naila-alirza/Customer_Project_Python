#!/usr/bin/env python
# coding: utf-8

# In[354]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[355]:


customers=pd.read_csv(r'C:\Users\HP\Downloads\Customers.txt')
customers.head()


# In[356]:


customers.info()


# In[357]:


customers.isnull().sum()


# In[358]:


customers.duplicated().sum()


# In[359]:


#drop email,adress, avatar columns because they will not help us to find the solution
customers.drop(columns=['Email','Address','Avatar'],axis=1,inplace=True)
customers.head()


# In[360]:


#outlier detection

for column in customers:
        plt.figure()
        sns.boxplot(data=numeric_custom, x=column)


# In[361]:


#outlier treatment
Q1 = customers.quantile(0.25)
Q3 = customers.quantile(0.75)
IQR = Q3 - Q1


# In[362]:


lower=Q1 - 1.5 * IQR
higher=Q3 + 1.5 * IQR


# In[363]:


customers = customers[~((customers < (lower)) |(customers > (higher))).any(axis=1)]
customers.head()


# In[364]:


#lets identify the correlations between features


# In[365]:


corr_matrix = customers.corr()
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[366]:


# We can see that there is strong positive correlation (0.77) between 'Yearly Amount Spent' and 'Length of Membership'
#lets visualize specially


# In[367]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers);


# In[368]:


# correlation between  time on web or app  and yearly amount  


# In[369]:


plt.scatter(x='Time on App',y='Yearly Amount Spent',data =customers)
plt.show()


# In[370]:


plt.scatter(x='Time on Website',y='Yearly Amount Spent',data =customers)
plt.show()


# In[371]:


# while there is no coorelation between time on website and yearly amount spent,correlation is better for app


# In[372]:


X = customers[customers.columns[:-1]]
y=customers['Yearly Amount Spent']


# In[373]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[374]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[375]:


lr.coef_


# In[376]:


predicted=lr.predict(X_test)


# In[377]:


plt.scatter(y_test,predicted)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# In[378]:


metrics.r2_score(y_test,predicted)


# In[379]:


metrics.mean_squared_error(y_test,predicted)


# In[380]:


coef = pd.DataFrame(lr.coef_,X.columns,columns= ['Coeff'])
coef


# In[381]:


#According to coefficinet and correlation  results,we can say customers spent time on app orders more (based on yearly amount) 
#than website.Length of Membership has big impact on decision making as well. Company should focus more effort on attracting 
#customers'attention for spending time  on website if it wants website using to catch up app using. 
#But company may have another approach and make more effort for app improvement because it is already certain 
#that customers like to spend time more on mobile app . 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




