
# coding: utf-8

# In[4]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__)) 
print('Sklearn: {}'.format(sklearn.__version__))      
      


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data=pd.read_csv('creditcard.csv')


# In[7]:


print(data.columns)


# In[11]:


data = data.sample(frac=0.1,random_state=1)
print(data.shape)
print(data.describe())


# In[12]:


print(data.shape)


# In[15]:


data.hist(figsize = (25,25))
plt.show()


# In[31]:


Fraud = data[data['class'] == 1]
Valid = data[data['class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))
print(outlier_fraction)

print('Fraud cases: {}'.format(len(Fraud)))
print('Valid cases: {}'.format(len(Valid)))


# In[26]:


corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()


# In[38]:


columns = data.columns.tolist()

target = "class"

X = data[columns]
Y = data[target]

print(X.shape)
print(y.shape)

