#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sensor_data = pd.read_csv("data/data_namuk/sensor_gun_all.csv")
sensor_data = sensor_data.rename(columns = {'Unnamed: 0': 'index'})
sensor_data = sensor_data.set_index('index')


# In[4]:


sensor_data['label'].value_counts()


# In[7]:


#print(sensor_data.isna().sum())


# In[8]:


sensor_data_target = sensor_data['label']
sensor_data_drop=sensor_data.drop('label', axis=1)
sensor_data_drop.head()


# In[9]:


sensor_data_drop.describe()


# In[10]:


Xtrain, X_test, y_train, y_test = train_test_split(sensor_data_drop, sensor_data_target, test_size=0.3, random_state=121)


# In[11]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(Xtrain, y_train)


# In[12]:


pred=dt_clf.predict(X_test)
print('예측정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))


# In[13]:


Xtrain.columns


# In[14]:


#특성값 중 상위 20위 확인
ftr_importances_values=dt_clf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=Xtrain.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20, y = ftr_top20.index)


# In[17]:


#평가
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    #precision = precision_score(y_test,pred)
    #recall = recall_score(y_test,pred)
    #f1 = f1_score(y_test,pred)
    print('오차행렬')
    print(confusion)
    #print('정확도:{0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f},f1:{3:.4f}'.format(accuracy,precision,recall))
    print('정확도:{0:.4f}'.format(accuracy))
    


# In[18]:


get_clf_eval(y_test, pred)


# In[20]:


sns.catplot(x= sensor_data['label'], y=sensor_data['LP_skew'],data=tips )


# In[34]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import export_graphviz


# In[48]:


export_graphviz(dt_clf, out_file='tree.dot', impurity=True, filled=True) 
#class_names=iris_data.target_names,feature_names=iris_data.feature_names에 대한 속성확인 후 적용


# In[49]:


#그림이 너무 커서 비주얼스튜디오로 그려야함.
import graphviz
with open('tree.dot') as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)

