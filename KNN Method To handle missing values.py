#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your data
data = pd.read_csv(r'C:\Users\DELL\Desktop\python_plfs.csv')


# Summary of the model
print(model.summary())


# In[3]:


df = pd.read_csv(r'C:\Users\DELL\Desktop\Book_PLFS.csv')
df.head()


# In[4]:


df.info()


# In[5]:


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Create the dataset from the given data
data = {
    'Date': [
        '01-04-2018', '01-07-2018', '01-10-2018', '01-01-2019',
        '01-04-2019', '01-07-2019', '01-10-2019', '01-01-2020',
        '01-04-2020', '01-07-2020', '01-10-2020', '01-01-2021',
        '01-04-2021', '01-07-2021', '01-10-2021', '01-01-2022',
        '01-04-2022', '01-07-2022', '01-10-2022', '01-01-2023',
        '01-04-2023', '01-07-2023', '01-10-2023', '01-01-2024'
    ],
    'Employment Rate': [
        46.2, 42.2, 42.2, 42.2, 42.4, 43.4, 44.1, 43.7, 
        np.nan, np.nan, np.nan, np.nan, np.nan, 42.3, 43.2, 
        43.4, 43.9, np.nan, 44.7, 45.2, 45.5, np.nan, 46.6, 46.9
    ]
}

df = pd.DataFrame(data)

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Initialize KNNImputer
imputer = KNNImputer(n_neighbors=4)

# Fit and transform the data
df[['Employment Rate']] = imputer.fit_transform(df[['Employment Rate']])

print(df)


# In[ ]:




