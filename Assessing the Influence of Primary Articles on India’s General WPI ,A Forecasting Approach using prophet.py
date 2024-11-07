#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install prophet')
get_ipython().system('pip install xgboost')
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import seaborn as sns
import pandas as pd


# In[2]:


file_path = r"C:\Users\DELL\Desktop\NAHI RUKNA 2.csv"
df = pd.read_csv(file_path)

# Print column names to check for unexpected spaces or names
print(df.columns)
df.head()


# In[3]:


df = pd.read_csv(r"C:\Users\DELL\Desktop\NAHI RUKNA 2.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# In[4]:


df.info()


# In[5]:


df.columns=['ds','y']


# In[6]:


# Convert 'ds' to datetime format
df['ds'] = pd.to_datetime(df['ds'], format='%d-%m-%Y')  # Adjust format if needed

# Check DataFrame structure
print(df.head())
print(df.columns)
print(df.dtypes)


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = r"C:\Users\DELL\Desktop\NAHI RUKNA 2.csv"
df = pd.read_csv(file_path)

# Print column names to check for unexpected spaces or names
print(df.columns)
print(df.head())

# Rename columns if necessary
df.columns = ['ds', 'y']

# Convert 'ds' to datetime format
df['ds'] = pd.to_datetime(df['ds'], format='%d-%m-%Y')  # Adjust format if needed

# Check DataFrame structure
print(df.head())
print(df.columns)
print(df.dtypes)

# Plotting
plt.figure(figsize=(18, 6))
plt.plot(df['ds'], df['y'], marker='o', linestyle='-', color='purple', alpha=0.7)
plt.title('Time Series Plot')
plt.xlabel('Year')
plt.ylabel('WPI')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels if needed for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[8]:


len(df)


# In[9]:


train = df.iloc[:len(df)-12]
test = df.iloc[len(df)-12:]


# In[10]:


m = Prophet()
m.fit(train)

# Create future dataframe
future = m.make_future_dataframe(periods=25, freq='M')

# Predict using the model
forecast = m.predict(future)


# In[11]:


forecast.tail()


# In[12]:


forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12)


# In[13]:


forecast[['yhat']].tail(12)


# In[14]:


test.tail(12)


# In[15]:


fig=plot_plotly(m,forecast)
fig.update_traces(line=dict(color='purple'))  # Set line color to purple


# In[16]:


fig=plot_components_plotly(m,forecast)
fig.update_traces(line=dict(color='purple'))  # Set line color 


# In[17]:


from statsmodels.tools.eval_measures import rmse


# In[18]:


predictions=forecast.iloc[-12]['yhat']
print("rmse:",rmse(predictions,test['y']))
print("mean:",test['y'].mean())


# # GENERAL WPI

# In[690]:


import pandas as pd

# Load DataFrame
file_path = r'C:\Users\DELL\Desktop\wpi python.csv'
df = pd.read_csv(file_path)

# Print column names to check for unexpected spaces or names
print(df.columns)
df.head()


# In[691]:


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# In[692]:


df.info()


# In[693]:


df.columns=['ds','y']


# In[694]:



# Convert 'ds' to datetime format
df['ds'] = pd.to_datetime(df['ds'], format='%d-%m-%Y')  # Adjust format if needed

# Check DataFrame structure
print(df.head())
print(df.columns)
print(df.dtypes)


# In[695]:


plt.figure(figsize=(18, 6))
plt.plot(df['ds'], df['y'], marker='o', linestyle='-', color='green', alpha=0.7)
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels if needed for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[696]:


len(df)


# In[697]:


train = df.iloc[:len(df)-32]
test = df.iloc[len(df)-32:]


# In[698]:


m = Prophet()
m.fit(train)

# Create future dataframe
future = m.make_future_dataframe(periods=45, freq='M')

# Predict using the model
forecast = m.predict(future)


# In[699]:


forecast.tail()


# In[700]:


forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12)


# In[708]:


forecast[['yhat']].tail(12)


# In[701]:


test.tail(12)


# In[707]:


fig=plot_plotly(m,forecast)
fig.update_traces(line=dict(color='green'))  # Set line color to purple


# In[706]:


fig=plot_components_plotly(m,forecast)
fig.update_traces(line=dict(color='green'))  # Set line color 


# In[704]:


from statsmodels.tools.eval_measures import rmse


# In[705]:


predictions=forecast.iloc[-32]['yhat']
print("rmse:",rmse(predictions,test['y']))
print("mean:",test['y'].mean())


# In[711]:


import matplotlib.pyplot as plt

# Data for Primary Articles Inflation Rates
months = [
    'July 2024', 'August 2024', 'September 2024', 'October 2024',
    'November 2024', 'December 2024', 'January 2025', 'February 2025',
    'March 2025', 'April 2025', 'May 2025', 'June 2025'
]

primary_articles_inflation = [
    6.36, 11.74, 11.09, 10.04, 12.65, 12.10, 11.44, 13.95,
    12.14, 11.66, 13.14, 9.65
]

general_wpi_inflation = [
    3.09, 3.29, 4.83, 4.62, 4.45, 5.07, 4.73, 5.52,
    6.53, 6.91, 6.41, 6.29
]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(months, primary_articles_inflation, marker='o', color='purple', label='Primary Articles Inflation')
plt.plot(months, general_wpi_inflation, marker='o', color='red', label='General WPI Inflation')

# Adding titles and labels
plt.title('Inflation Rates Comparison: Primary Articles vs General WPI')
plt.xlabel('Month')
plt.ylabel('Inflation Rate (%)')
plt.xticks(rotation=45)
plt.legend()


# Show plot
plt.tight_layout()
plt.show()


# In[712]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data for Primary Articles Inflation Rates
months = [
    'July 2024', 'August 2024', 'September 2024', 'October 2024',
    'November 2024', 'December 2024', 'January 2025', 'February 2025',
    'March 2025', 'April 2025', 'May 2025', 'June 2025'
]

primary_articles_inflation = [
    6.36, 11.74, 11.09, 10.04, 12.65, 12.10, 11.44, 13.95,
    12.14, 11.66, 13.14, 9.65
]

general_wpi_inflation = [
    3.09, 3.29, 4.83, 4.62, 4.45, 5.07, 4.73, 5.52,
    6.53, 6.91, 6.41, 6.29
]

# Convert months to numerical values for interpolation
x = np.arange(len(months))

# Create spline interpolation for smooth lines
xnew = np.linspace(x.min(), x.max(), 300)  # 300 points for smooth curve

spl_primary = make_interp_spline(x, primary_articles_inflation, k=3)
spl_general_wpi = make_interp_spline(x, general_wpi_inflation, k=3)

primary_articles_smooth = spl_primary(xnew)
general_wpi_smooth = spl_general_wpi(xnew)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(xnew, primary_articles_smooth, color='blue', label='Primary Articles Inflation', linestyle='-', linewidth=2)
plt.plot(xnew, general_wpi_smooth, color='red', label='General WPI Inflation', linestyle='-', linewidth=2)

# Adding titles and labels
plt.title('Inflation Rates Comparison: Primary Articles vs General WPI')
plt.xlabel('Month')
plt.ylabel('Inflation Rate (%)')
plt.xticks(ticks=x, labels=months, rotation=45)
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




