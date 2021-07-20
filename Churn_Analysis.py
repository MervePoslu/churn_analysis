#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd


# In[47]:


import numpy as np


# In[48]:


import matplotlib.pyplot as plt 


# In[49]:


import seaborn as sns 
from scipy import stats 
import matplotlib.ticker as mtick # For specifying the axes tick format 
import matplotlib.pyplot as plt


# In[50]:


df = pd.read_excel('/Users/merve.poslu/Desktop/pythonProject/Telco-Customer-Churn.xlsx')
                   
df.head(5)


# In[11]:


df.info() 


# In[51]:


df.describe()


# In[52]:



df.loc[df.Churn=='No','Churn'] = 0 
df.loc[df.Churn=='Yes','Churn'] = 1


# In[53]:


df.head()


# In[54]:


dataset = df['Churn'].value_counts()


# In[55]:


dataset


# In[56]:


df1 = pd.read_excel('/Users/merve.poslu/Desktop/pythonProject/Telco-Customer-Churn.xlsx')
df.head()


# In[57]:


char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
for c in char_cols:
    df[c] = pd.factorize(df[c])[0]
df.head()


# In[58]:


colors = ['#4D3425','#E4512B']
ax = (df['gender'].value_counts()*100.0 /len(df)).plot(kind='bar',
                                           stacked = True,
                                               rot = 0,
                                            color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers')
ax.set_xlabel('Gender')
ax.set_ylabel('% Customers')
ax.set_title('Gender Distribution')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-3.5,             str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold')


# In[59]:


sizes = [5174,1869]
labels='NO','YES'
explode = (0, 0.1)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode,autopct='%1.1f%%',shadow=True, startangle=75 )
ax1.axis('equal') 
ax1.set_title("Client Churn Distribution")

ax1.legend(labels)

plt.show()


# In[25]:


df.groupby('gender').Churn.mean() 


# In[60]:


Churn_Mean = [0.269209, 0.261603]
Gender = ('Female', 'Male')
x_pos = np.arange(len(Churn_Mean))
plt.bar(x_pos, Churn_Mean, color=['orange','blue'])

# Create names on the x-axis
plt.xticks(x_pos, Gender)

# Add title and axis names
#plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate')
#plt.legend()

# Show graph
plt.show()


# In[61]:


catvars = df1.columns.tolist()
catvars = [e for e in catvars if e not in ('TotalCharges', 'MonthlyCharges', 
                                           'tenure', 'customerID', 'Churn')]

y = 'Churn'
for x in catvars:
    plot = df1.groupby(x)[y]        .value_counts(normalize=True).mul(100)        .rename('percent').reset_index()        .pipe((sns.catplot,'data'), x=x, y='percent', hue=y, kind='bar')
    plot.fig.suptitle("Churn by " + x)
    plot


# In[62]:


sns.distplot(df1.tenure) 


# In[63]:


#Churn by tenure 
bins = 30
plt.hist(df1[df1.Churn == 'Yes'].tenure, 
         bins, alpha=0.5, density=True, label='Churned')
plt.hist(df1[df1.Churn == 'No'].tenure, 
         bins, alpha=0.5, density=True, label="Didn't Churn")
plt.legend(loc='upper right')
plt.show()


# In[64]:


churners_number = len(df[df['Churn'] == 1])
print("Number of churners", churners_number)

churners = (df[df['Churn'] == 1])

non_churners = df[df['Churn'] == 0].sample(n=churners_number)
print("Number of non-churners", len(non_churners))
df3 = churners.append(non_churners)


# In[65]:


def show_correlations(df, show_chart = True):
    fig = plt.figure(figsize = (20,10))
    corr = df.corr()
    if show_chart == True:
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True)
    return corr


# In[66]:


correlation_df = show_correlations(df3,show_chart = True)


# In[67]:


df = pd.get_dummies(data=df)
df.head()


# In[68]:


# Define the target variable (dependent variable) 
y = df.Churn 
df = df.drop(['Churn'], axis= 1)


# In[69]:


df


# In[71]:


# Splitting training and testing data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.20) 


# In[ ]:


# Applying Support Vector Machine algorithm
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', degree=8)  
svclassifier.fit(X_train, y_train)


# In[ ]:


SVC(degree=8, kernel='linear')


# In[48]:


# Predicting part, applying the model to predict
y_pred = svclassifier.predict(X_test) 


# In[49]:


# Evaluating model performance
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

