#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset  =  pd.read_csv('titanic_train.csv')


# In[3]:


dataset.head(20)


# In[4]:


dataset.info()


# In[5]:


dataset.columns


# In[10]:


import seaborn as sns
sns.set()


# In[11]:


gender = dataset['Sex']


# In[12]:


# bar graph
sns.countplot(gender)


# In[13]:


sns.countplot(dataset['Survived'], hue='Sex', data=dataset)


# In[14]:


sns.countplot(dataset['Survived'], hue='Pclass', data=dataset)


# In[15]:


age = dataset['Age']


# In[16]:


sns.distplot(age)


# In[17]:


type(dataset)


# In[18]:


dataset.isnull()


# In[ ]:





# In[19]:


sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')


# In[20]:


dataset.head(10)


# In[21]:


sns.distplot(age.dropna() ,bins=40)


# In[22]:


dataset.columns


# In[23]:


sns.countplot(dataset['SibSp'], data=dataset, hue='Survived')


# In[24]:


dataset.columns


# In[25]:


age.mean()


# In[26]:


dataset


# In[27]:


sns.boxplot(data=dataset, y='Age' , x='Pclass')


# In[28]:


age


# In[29]:


def lw(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        elif Pclass == 3:
            return 25
        else:
            return 30
    else:
        return age
    


# In[30]:


dataset['Age'] = dataset[['Age', 'Pclass']].apply(lw , axis=1)


# In[31]:


dataset


# In[32]:


sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')


# In[33]:


dataset.drop('Cabin', axis=1, inplace=True )


# In[34]:


dataset


# In[35]:


sns.heatmap(dataset.isnull(), cbar=False, yticklabels=False , cmap='viridis')


# In[36]:


dataset


# In[37]:


type(dataset)


# In[38]:


# univariate : histogram : frequency distribution
fare = dataset['Fare']


# In[39]:


type(fare)


# In[40]:


fare.hist(bins=50, color='red', figsize=(5,1) )


# In[41]:


# fare.plot()


# In[42]:


y = dataset['Survived']


# In[43]:


dataset.columns


# In[44]:


X = dataset[ ['Pclass','Sex', 'Age', 'SibSp', 'Parch' , 'Embarked' ]]


# In[45]:


# string : number(label encoding) :  dummy varaible: one hot encoding, remove one dummy varaible

sex = dataset['Sex']


# In[46]:


sex = pd.get_dummies(sex, drop_first=True )


# In[47]:


pclass = dataset['Pclass']


# In[48]:


pclass = pd.get_dummies(pclass, drop_first=True)


# In[49]:


sibsp = dataset['SibSp']


# In[50]:


sibsp = pd.get_dummies(sibsp, drop_first=True)


# In[51]:


parch = dataset['Parch']
parch = pd.get_dummies(parch, drop_first=True)


# In[52]:


embarked = dataset['Embarked']
embarked = pd.get_dummies(embarked, drop_first=True)


# In[53]:


age = dataset[ 'Age']


# In[54]:


type(X)


# In[55]:


type(parch)


# In[56]:


X = pd.concat([age, embarked, parch, sibsp, pclass, sex] ,  axis=1)


# In[57]:


X


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


model = LogisticRegression()


# In[63]:


model.fit(X_train,y_train)


# In[64]:


model.coef_


# In[65]:


y_pred = model.predict(X_test)


# In[67]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test , y_pred)


# In[70]:


record = 141+16+35+76


# In[71]:


record


# In[72]:


trueanswer = 141 + 76


# In[73]:


trueanswer


# In[74]:


accuracy = trueanswer / record * 100


# In[75]:


accuracy


# In[76]:


error = 35 + 16


# In[77]:


error / record * 100


# In[78]:


from sklearn.metrics import classification_report


# In[79]:


print(classification_report(y_test, y_pred))

accuracy = int(accuracy)
f = open("./accuracy.txt", "w+")
f.write(str(accuracy))
f.close()

print("Accuracy = ", accuracy , "%")





