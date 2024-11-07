#!/usr/bin/env python
# coding: utf-8

# # Male employment Rate from April-june 2018 to jan-march 2024 in urban area for major states and india.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Updated data in tabular format
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [68.3, 67.5, 68, 68.9, 68.2, 69.3, 69.9, 68.7, 45.4, 58.3, 52.2, 64.7, 62.4, 66.5, 68.3, 68.6, 69.5, 70.6, 69.2, 71.1, 70.8, 70.5, 70.4, 70],
    "Gujarat": [72.4, 73, 73, 72.1, 74.7, 74.2, 76.2, 76.4, 67.4, 73.6, 73.9, 75.4, 73.9, 74.3, 75, 74.8, 76.5, 77, 77.2, 76.3, 75.7, 76, 75.4, 75.3],
    "Madhya Pradesh": [64.8, 67.3, 67.1, 63.9, 62.8, 63.2, 63.7, 64.2, 52.3, 62.9, 66.9, 67, 62.3, 68.5, 68.1, 66.2, 65.6, 67.3, 68.1, 69.8, 70.2, 71.7, 72.1, 72.2],
    "Uttar Pradesh": [61.5, 63.4, 63, 62.9, 64.3, 65.5, 65.7, 65.4, 57.7, 63.2, 65.9, 66, 63, 65.1, 66.7, 66.2, 67.9, 68, 67.8, 68.4, 68.8, 69.5, 69.9, 70],
    "West Bengal": [68.4, 67.5, 67.7, 69.7, 70.9, 71.4, 72.6, 70.8, 60.5, 67.2, 71.1, 71.1, 66.6, 68.7, 69.5, 71.4, 72.1, 72.9, 73, 73.4, 73.4, 72.9, 73.4, 74.2],
    "Tamil Nadu": [68.6, 67.5, 67.7, 67.5, 67.1, 69.3, 69.2, 68.1, 59.3, 67.6, 69.4, 69.1, 64, 67.5, 67.7, 69.3, 68.2, 67.9, 68.4, 68.4, 67.9, 68.3, 67.9, 68],
    "India": [67, 67, 66.9, 67, 67.3, 68, 68.4, 67.3, 56.9, 64.3, 66.7, 67.2, 64.2, 66.6, 67.8, 67.7, 68.3, 68.6, 68.6, 69.1, 69.2, 69.4, 69.8, 69.8]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Converting 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Plotting the line graph
plt.figure(figsize=(12, 8))
for column in df.columns[1:]:
    plt.plot(df['Date'], df[column], label=column)

plt.xlabel('Year')
plt.ylabel('Employment Rate(%)')
plt.ylim(0,100)
#plt.title('Time plot of quarterly urban UR estimates for Males of age 15+ years from April-June,  2018 to January-march, 2024 for the major Indian states as well as at all Indi')
plt.legend()
plt.grid(True)
plt.show()


# # feMale employment Rate from April-june 2018 to jan-march 2024 in urban area for major states and india.

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Updated data in tabular format
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [17.2, 17.3, 18.7, 19.6, 19, 21.8, 22.1, 22.7, 13.7, 16, 22.2, 19.1, 18.3, 20.8, 22.2, 22.6, 23, 24, 24, 24.2, 24.3, 24.2, 24.9, 26.1],
    "Gujarat": [14.8, 16.2, 14.5, 15, 15.6, 16.2, 18, 18.4, 15.2, 17.8, 18.4, 19.2, 16.2, 14.8, 16.5, 15.7, 18.4, 20.6, 20.7, 23.8, 24.4, 24.5, 25.5, 25.2],
    "Madhya Pradesh": [15.5, 16.4, 15.4, 14.3, 13.2, 15.4, 16.9, 17.4, 12.6, 15.1, 15.7, 16.6, 14.5, 14.7, 15.8, 15, 15.6, 16.5, 16.5, 17.2, 15.4, 18.8, 22.1, 22],
    "Uttar Pradesh": [8.9, 8.8, 8.5, 7.1, 8.5, 8.8, 9.7, 10.7, 10.1, 10.1, 10.2, 9.7, 10.6, 9.5, 9.6, 9.1, 9.7, 9.8, 10.8, 11.9, 11.6, 13, 14.1, 13.5],
    "West Bengal": [20.9, 22.7, 21.8, 22.2, 22.2, 22.9, 23.6, 22.4, 19, 21.7, 22.2, 22.3, 20, 17.9, 19.8, 21.8, 22.4, 24.1, 25.6, 23.7, 23.9, 24.3, 25.4, 28],
    "Tamil Nadu": [22, 22.7, 23.1, 23.8, 23.6, 27.2, 27, 27.7, 23.3, 26, 26.3, 26.9, 22.9, 24.8, 23.2, 24.4, 24.9, 24.1, 25.6, 25, 24.3, 23.6, 23.7, 24.5],
    "India": [16.4, 17.1, 17.2, 16.9, 16.9, 18.3, 19, 19.6, 15.5, 17.1, 17.9, 18.7, 17.2, 17.6, 18.1, 18.3, 18.9, 19.7, 20.2, 20.6, 21.1, 21.9, 22.9, 23.4]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Converting 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Plotting the line graph
plt.figure(figsize=(12, 8))
for column in df.columns[1:]:
    plt.plot(df['Date'], df[column], label=column)

plt.xlabel('Year')
plt.ylabel('Employment Rate(%)')
#plt.title('Time Series Data of Different States')
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.show()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

# Updated data in tabular format
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [68.3, 67.5, 68, 68.9, 68.2, 69.3, 69.9, 68.7, 45.4, 58.3, 52.2, 64.7, 62.4, 66.5, 68.3, 68.6, 69.5, 70.6, 69.2, 71.1, 70.8, 70.5, 70.4, 70],
    "Gujarat": [72.4, 73, 73, 72.1, 74.7, 74.2, 76.2, 76.4, 67.4, 73.6, 73.9, 75.4, 73.9, 74.3, 75, 74.8, 76.5, 77, 77.2, 76.3, 75.7, 76, 75.4, 75.3],
    "Madhya Pradesh": [64.8, 67.3, 67.1, 63.9, 62.8, 63.2, 63.7, 64.2, 52.3, 62.9, 66.9, 67, 62.3, 68.5, 68.1, 66.2, 65.6, 67.3, 68.1, 69.8, 70.2, 71.7, 72.1, 72.2],
    "Uttar Pradesh": [61.5, 63.4, 63, 62.9, 64.3, 65.5, 65.7, 65.4, 57.7, 63.2, 65.9, 66, 63, 65.1, 66.7, 66.2, 67.9, 68, 67.8, 68.4, 68.8, 69.5, 69.9, 70],
    "West Bengal": [68.4, 67.5, 67.7, 69.7, 70.9, 71.4, 72.6, 70.8, 60.5, 67.2, 71.1, 71.1, 66.6, 68.7, 69.5, 71.4, 72.1, 72.9, 73, 73.4, 73.4, 72.9, 73.4, 74.2],
    "Tamil Nadu": [68.6, 67.5, 67.7, 67.5, 67.1, 69.3, 69.2, 68.1, 59.3, 67.6, 69.4, 69.1, 64, 67.5, 67.7, 69.3, 68.2, 67.9, 68.4, 68.4, 67.9, 68.3, 67.9, 68],
    "India": [67, 67, 66.9, 67, 67.3, 68, 68.4, 67.3, 56.9, 64.3, 66.7, 67.2, 64.2, 66.6, 67.8, 67.7, 68.3, 68.6, 68.6, 69.1, 69.2, 69.4, 69.8, 69.8]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Converting 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Plotting the line graph
plt.figure(figsize=(12, 8))
for column in df.columns[1:]:
    plt.plot(df['Date'], df[column], label=column)

plt.xlabel('Date')
plt.ylabel('Values')
plt.ylim(0,100)
plt.title('Time plot of quarterly urban UR estimates for Males of age 15+ years from April-June,  2018 to January-march, 2024 for the major Indian states as well as at all Indi')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Updated data in tabular format
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [17.2, 17.3, 18.7, 19.6, 19, 21.8, 22.1, 22.7, 13.7, 16, 22.2, 19.1, 18.3, 20.8, 22.2, 22.6, 23, 24, 24, 24.2, 24.3, 24.2, 24.9, 26.1],
    "Gujarat": [14.8, 16.2, 14.5, 15, 15.6, 16.2, 18, 18.4, 15.2, 17.8, 18.4, 19.2, 16.2, 14.8, 16.5, 15.7, 18.4, 20.6, 20.7, 23.8, 24.4, 24.5, 25.5, 25.2],
    "Madhya Pradesh": [15.5, 16.4, 15.4, 14.3, 13.2, 15.4, 16.9, 17.4, 12.6, 15.1, 15.7, 16.6, 14.5, 14.7, 15.8, 15, 15.6, 16.5, 16.5, 17.2, 15.4, 18.8, 22.1, 22],
    "Uttar Pradesh": [8.9, 8.8, 8.5, 7.1, 8.5, 8.8, 9.7, 10.7, 10.1, 10.1, 10.2, 9.7, 10.6, 9.5, 9.6, 9.1, 9.7, 9.8, 10.8, 11.9, 11.6, 13, 14.1, 13.5],
    "West Bengal": [20.9, 22.7, 21.8, 22.2, 22.2, 22.9, 23.6, 22.4, 19, 21.7, 22.2, 22.3, 20, 17.9, 19.8, 21.8, 22.4, 24.1, 25.6, 23.7, 23.9, 24.3, 25.4, 28],
    "Tamil Nadu": [22, 22.7, 23.1, 23.8, 23.6, 27.2, 27, 27.7, 23.3, 26, 26.3, 26.9, 22.9, 24.8, 23.2, 24.4, 24.9, 24.1, 25.6, 25, 24.3, 23.6, 23.7, 24.5],
    "India": [16.4, 17.1, 17.2, 16.9, 16.9, 18.3, 19, 19.6, 15.5, 17.1, 17.9, 18.7, 17.2, 17.6, 18.1, 18.3, 18.9, 19.7, 20.2, 20.6, 21.1, 21.9, 22.9, 23.4]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Converting 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Plotting the line graph
plt.figure(figsize=(12, 8))
for column in df.columns[1:]:
    plt.plot(df['Date'], df[column], label=column)

plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Time Series Data of Different States')
plt.ylim(0,100)
plt.legend()
plt.grid(True)
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Male and Female data
data = {
    "Year": ["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "Maharashtra_Male": [68.3, 68.9, 45.4, 64.7, 69.5, 70.8, 70],
    "Maharashtra_Female": [17.2, 19.6, 13.7, 19.1, 23, 24.3, 26.1],
    "Gujarat_Male": [72.4, 72.1, 67.4, 75.4, 76.5, 75.7, 75.3],
    "Gujarat_Female": [14.8, 15, 15.2, 19.2, 18.4, 24.4, 25.2],
    "Madhya Pradesh_Male": [64.8, 63.9, 52.3, 67, 65.6, 70.2, 72.2],
    "Madhya Pradesh_Female": [15.5, 14.3, 12.6, 16.6, 15.6, 15.4, 22],
    "Uttar Pradesh_Male": [61.5, 62.9, 57.7, 66, 67.9, 68.8, 70],
    "Uttar Pradesh_Female": [8.9, 7.1, 10.1, 9.7, 9.7, 11.6, 13.5],
    "West Bengal_Male": [68.4, 69.7, 60.5, 71.1, 72.1, 73.4, 74.2],
    "West Bengal_Female": [20.9, 22.2, 19, 22.3, 22.4, 23.9, 28],
    "Tamil Nadu_Male": [68.6, 67.5, 59.3, 69.1, 68.2, 67.9, 68],
    "Tamil Nadu_Female": [22, 23.8, 23.3, 26.9, 24.9, 23.9, 25.5],
    "India_Male": [67, 67, 56.9, 67.2, 68.3, 69.2, 69.8],
    "India_Female": [15.9, 16.1, 14.5, 17.5, 18.1, 19.2, 21.2]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt the data for plotting
df_melted = pd.melt(df, id_vars=["Year"], var_name="State_Gender", value_name="Employment Rate")

# Split 'State_Gender' into separate 'State' and 'Gender' columns
df_melted[['State', 'Gender']] = df_melted['State_Gender'].str.split('_', expand=True)

# Plot
plt.figure(figsize=(14, 8))
states = df_melted['State'].unique()
years = df_melted['Year'].unique()
bar_width = 0.35
index = range(len(states))

for i, year in enumerate(years):
    male_data = df_melted[(df_melted['Year'] == year) & (df_melted['Gender'] == 'Male')]
    female_data = df_melted[(df_melted['Year'] == year) & (df_melted['Gender'] == 'Female')]
    
    plt.bar([x + i * bar_width for x in index], male_data['Employment Rate'], bar_width, alpha=0.6, label=f'{year} Male')
    plt.bar([x + i * bar_width + bar_width/2 for x in index], female_data['Employment Rate'], bar_width, alpha=0.6, label=f'{year} Female')

plt.xlabel("State")
plt.ylabel("Employment Rate (%)")
plt.title("Urban Workforce Participation Rate by Gender and State (2018-2024)")
plt.xticks([x + (len(years) * bar_width) / 2 for x in index], states)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# # Applying Grey Model to predict next 2 quaters

# In[5]:


import numpy as np
import pandas as pd

# Function to perform GM(1,1) forecasting
def gm11(x0, n_predict):
    n = len(x0)
    x1 = np.cumsum(x0)  # Cumulative sum of the original data
    
    # Coefficient estimation
    B = np.array([-0.5 * (x1[1:] + x1[:-1]), np.ones(n-1)]).T
    Y = x0[1:]
    coeff = np.linalg.inv(B.T @ B) @ B.T @ Y
    
    a, b = coeff[0], coeff[1]
    
    # Predict the values
    x_hat = np.zeros(n + n_predict)
    x_hat[0] = x0[0]
    
    for k in range(1, n + n_predict):
        x_hat[k] = (x0[0] - b/a) * np.exp(-a * k) + b/a
    
    # Calculate the non-cumulative predicted values
    x0_hat = np.diff(x_hat)
    
    return x0_hat[:n], x0_hat[n:]  # First element is the fit, second is the forecast

# Function to compute RMAPE
def rmape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Load the data (assuming it's already loaded in DataFrame `df`)
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [68.3, 67.5, 68, 68.9, 68.2, 69.3, 69.9, 68.7, 45.4, 58.3, 52.2, 64.7, 62.4, 66.5, 68.3, 68.6, 69.5, 70.6, 69.2, 71.1, 70.8, 70.5, 70.4, 70],
    "Gujarat": [72.4, 73, 73, 72.1, 74.7, 74.2, 76.2, 76.4, 67.4, 73.6, 73.9, 75.4, 73.9, 74.3, 75, 74.8, 76.5, 77, 77.2, 76.3, 75.7, 76, 75.4, 75.3],
    "Madhya Pradesh": [64.8, 67.3, 67.1, 63.9, 62.8, 63.2, 63.7, 64.2, 52.3, 62.9, 66.9, 67, 62.3, 68.5, 68.1, 66.2, 65.6, 67.3, 68.1, 69.8, 70.2, 71.7, 72.1, 72.2],
    "Uttar Pradesh": [61.5, 63.4, 63, 62.9, 64.3, 65.5, 65.7, 65.4, 57.7, 63.2, 65.9, 66, 63, 65.1, 66.7, 66.2, 67.9, 68, 67.8, 68.4, 68.8, 69.5, 69.9, 70],
    "West Bengal": [68.4, 67.5, 67.7, 69.7, 70.9, 71.4, 72.6, 70.8, 60.5, 67.2, 71.1, 71.1, 66.6, 68.7, 69.5, 71.4, 72.1, 72.9, 73, 73.4, 73.4, 72.9, 73.4, 74.2],
    "Tamil Nadu": [68.6, 67.5, 67.7, 67.5, 67.1, 69.3, 69.2, 68.1, 59.3, 67.6, 69.4, 69.1, 64, 67.5, 67.7, 69.3, 68.2, 67.9, 68.4, 68.4, 67.9, 68.3, 67.9, 68],
    "India": [67, 67, 66.9, 67, 67.3, 68, 68.4, 67.3, 56.9, 64.3, 66.7, 67.2, 64.2, 66.6, 67.8, 67.7, 68.3, 68.6, 68.6, 69.1, 69.2, 69.4, 69.8, 69.8]
}
    
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Initialize a dictionary to store forecasts and RMAPE values
forecasts = {}
rmapes = {}

for column in df.columns[1:]:  # Skip the 'Date' column
    x0 = df[column].values
    n_predict = 2  # Forecasting for the next 2 quarters
    
    # Apply the Grey Model
    fit, forecast = gm11(x0, n_predict)
    
    # If the forecast length is less than expected, fill the remaining with the last forecasted value
    if len(forecast) < n_predict:
        forecast = np.append(forecast, [forecast[-1]] * (n_predict - len(forecast)))
    
    # Calculate RMAPE for the model on the actual fitted data
    fitted_values, _ = gm11(x0[:-2], n_predict=2)  # Remove last two points and fit again
    rmape_value = rmape(x0[:-2], fitted_values[:len(x0)-2])
    
    # Store results
    forecasts[column] = forecast
    rmapes[column] = rmape_value

# Extending the dates for the forecast
dates_extended = pd.date_range(start=df['Date'].iloc[-1], periods=n_predict+1, freq='Q')[1:]

# Creating the result DataFrame
forecast_df = pd.DataFrame({'Date': dates_extended})
for column in forecasts:
    forecast_df[column] = forecasts[column]

print("Forecasts for the next two quarters:")
print(forecast_df)
print("\nRMAPE values for each state:")
print(rmapes)


# In[4]:


import matplotlib.pyplot as plt

# Historical data
dates = ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
         "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
         "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"]

maharashtra = [68.3, 67.5, 68, 68.9, 68.2, 69.3, 69.9, 68.7, 45.4, 58.3, 52.2, 64.7, 62.4, 66.5, 68.3, 68.6, 69.5, 70.6, 69.2, 71.1, 70.8, 70.5, 70.4, 70]
gujarat = [72.4, 73, 73, 72.1, 74.7, 74.2, 76.2, 76.4, 67.4, 73.6, 73.9, 75.4, 73.9, 74.3, 75, 74.8, 76.5, 77, 77.2, 76.3, 75.7, 76, 75.4, 75.3]
madhya_pradesh = [64.8, 67.3, 67.1, 63.9, 62.8, 63.2, 63.7, 64.2, 52.3, 62.9, 66.9, 67, 62.3, 68.5, 68.1, 66.2, 65.6, 67.3, 68.1, 69.8, 70.2, 71.7, 72.1, 72.2]
uttar_pradesh = [61.5, 63.4, 63, 62.9, 64.3, 65.5, 65.7, 65.4, 57.7, 63.2, 65.9, 66, 63, 65.1, 66.7, 66.2, 67.9, 68, 67.8, 68.4, 68.8, 69.5, 69.9, 70]
west_bengal = [68.4, 67.5, 67.7, 69.7, 70.9, 71.4, 72.6, 70.8, 60.5, 67.2, 71.1, 71.1, 66.6, 68.7, 69.5, 71.4, 72.1, 72.9, 73, 73.4, 73.4, 72.9, 73.4, 74.2]
tamil_nadu = [68.6, 67.5, 67.7, 67.5, 67.1, 69.3, 69.2, 68.1, 59.3, 67.6, 69.4, 69.1, 64, 67.5, 67.7, 69.3, 68.2, 67.9, 68.4, 68.4, 67.9, 68.3, 67.9, 68]
india = [67, 67, 66.9, 67, 67.3, 68, 68.4, 67.3, 56.9, 64.3, 66.7, 67.2, 64.2, 66.6, 67.8, 67.7, 68.3, 68.6, 68.6, 69.1, 69.2, 69.4, 69.8, 69.8]

# Forecasted data
forecast_date = "01-04-2024"
forecast_maharashtra = 70.08104
forecast_gujarat = 76.703585
forecast_madhya_pradesh = 71.48153
forecast_uttar_pradesh = 70.263078
forecast_west_bengal = 73.949496
forecast_tamil_nadu = 68.199595
forecast_india = 69.452751

# Add forecasted values to the data
dates.append(forecast_date)
maharashtra.append(forecast_maharashtra)
gujarat.append(forecast_gujarat)
madhya_pradesh.append(forecast_madhya_pradesh)
uttar_pradesh.append(forecast_uttar_pradesh)
west_bengal.append(forecast_west_bengal)
tamil_nadu.append(forecast_tamil_nadu)
india.append(forecast_india)

# Plotting the data
plt.figure(figsize=(12, 8))

plt.plot(dates[:-1], maharashtra[:-1], marker='o', label='Maharashtra - Historical', color='blue')
plt.plot(dates[:-1], gujarat[:-1], marker='o', label='Gujarat - Historical', color='green')
plt.plot(dates[:-1], madhya_pradesh[:-1], marker='o', label='Madhya Pradesh - Historical', color='red')
plt.plot(dates[:-1], uttar_pradesh[:-1], marker='o', label='Uttar Pradesh - Historical', color='purple')
plt.plot(dates[:-1], west_bengal[:-1], marker='o', label='West Bengal - Historical', color='orange')
plt.plot(dates[:-1], tamil_nadu[:-1], marker='o', label='Tamil Nadu - Historical', color='brown')
plt.plot(dates[:-1], india[:-1], marker='o', label='India - Historical', color='black')

# Adding forecasted values as dotted lines
plt.plot(dates[-2:], maharashtra[-2:], 'b--', marker='o', label='Maharashtra - Forecasted', color='blue')
plt.plot(dates[-2:], gujarat[-2:], 'g--', marker='o', label='Gujarat - Forecasted', color='green')
plt.plot(dates[-2:], madhya_pradesh[-2:], 'r--', marker='o', label='Madhya Pradesh - Forecasted', color='red')
plt.plot(dates[-2:], uttar_pradesh[-2:], 'p--', marker='o', label='Uttar Pradesh - Forecasted', color='purple')
plt.plot(dates[-2:], west_bengal[-2:], 'o--', marker='o', label='West Bengal - Forecasted', color='orange')
plt.plot(dates[-2:], tamil_nadu[-2:], 'brown', linestyle='dashed', marker='o', label='Tamil Nadu - Forecasted')
plt.plot(dates[-2:], india[-2:], 'k--', marker='o', label='India - Forecasted', color='black')

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Employment Rate')
plt.title('Employment Rate by State and India (Historical and Forecasted)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# In[12]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Function to perform GM(1,1) forecasting
def gm11(x0, n_predict):
    n = len(x0)
    x1 = np.cumsum(x0)  # Cumulative sum of the original data
    
    # Coefficient estimation
    B = np.array([-0.5 * (x1[1:] + x1[:-1]), np.ones(n-1)]).T
    Y = x0[1:]
    coeff = np.linalg.inv(B.T @ B) @ B.T @ Y
    
    a, b = coeff[0], coeff[1]
    
    # Predict the values
    x_hat = np.zeros(n + n_predict)
    x_hat[0] = x0[0]
    
    for k in range(1, n + n_predict):
        x_hat[k] = (x0[0] - b/a) * np.exp(-a * k) + b/a
    
    # Calculate the non-cumulative predicted values
    x0_hat = np.diff(x_hat)
    
    return x0_hat[:n], x0_hat[n:]  # First element is the fit, second is the forecast

# Function to compute RMSE
def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Load the data (assuming it's already loaded in DataFrame `df`)
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [17.2, 17.3, 18.7, 19.6, 19.0, 21.8, 22.1, 22.7, 13.7, 16.0, 22.2, 19.1, 18.3, 20.8, 22.2, 22.6, 23.0, 24.0, 24.0, 24.2, 24.3, 24.2, 24.9, 26.1],
    "Gujarat": [14.8, 16.2, 14.5, 15.0, 15.6, 16.2, 18.0, 18.4, 15.2, 17.8, 18.4, 19.2, 16.2, 14.8, 16.5, 15.7, 18.4, 20.6, 20.7, 23.8, 24.4, 24.5, 25.5, 25.2],
    "Madhya Pradesh": [15.5, 16.4, 15.4, 14.3, 13.2, 15.4, 16.9, 17.4, 12.6, 15.1, 15.7, 16.6, 14.5, 14.7, 15.8, 15.0, 15.6, 16.5, 16.5, 17.2, 15.4, 18.8, 22.1, 22.0],
    "Uttar Pradesh": [8.9, 8.8, 8.5, 7.1, 8.5, 8.8, 9.7, 10.7, 10.1, 10.1, 10.2, 9.7, 10.6, 9.5, 9.6, 9.1, 9.7, 9.8, 10.8, 11.9, 11.6, 13.0, 14.1, 13.5],
    "West Bengal": [20.9, 22.7, 21.8, 22.2, 22.2, 22.9, 23.6, 22.4, 19.0, 21.7, 22.2, 22.3, 20.0, 17.9, 19.8, 21.8, 22.4, 24.1, 25.6, 23.7, 23.9, 24.3, 25.4, 28.0],
    "Tamil Nadu": [22.0, 22.7, 23.1, 23.8, 23.6, 27.2, 27.0, 27.7, 23.3, 26.0, 26.3, 26.9, 22.9, 24.8, 23.2, 24.4, 24.9, 24.1, 25.6, 25.0, 24.3, 23.6, 23.7, 24.5],
    "India": [16.4, 17.1, 17.2, 16.9, 16.9, 18.3, 19.0, 19.6, 15.5, 17.1, 17.9, 18.7, 17.2, 17.6, 18.1, 18.3, 18.9, 19.7, 20.2, 20.6, 21.1, 21.9, 22.9, 23.4]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Initialize a dictionary to store forecasts and RMSE values
forecasts = {}
rmses = {}

for column in df.columns[1:]:  # Skip the 'Date' column
    x0 = df[column].values
    n_predict = 2  # Forecasting for the next 2 quarters
    
    # Apply the Grey Model
    fit, forecast = gm11(x0, n_predict)
    
    # If the forecast length is less than expected, fill the remaining with the last forecasted value
    if len(forecast) < n_predict:
        forecast = np.append(forecast, [forecast[-1]] * (n_predict - len(forecast)))
    
    # Calculate RMSE for the model on the actual fitted data
    fitted_values, _ = gm11(x0[:-2], n_predict=2)  # Remove last two points and fit again
    rmse_value = rmse(x0[:-2], fitted_values[:len(x0)-2])
    
    # Store results
    forecasts[column] = forecast
    rmses[column] = rmse_value

# Extending the dates for the forecast
dates_extended = pd.date_range(start=df['Date'].iloc[-1], periods=n_predict+1, freq='Q')[1:]

# Creating the result DataFrame
forecast_df = pd.DataFrame({'Date': dates_extended})
for column in forecasts:
    forecast_df[column] = forecasts[column]

print("Forecasts for the next two quarters:")
print(forecast_df)
print("\nRMSE values for each state:")
print(rmses)


# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Function to perform GM(1,1) forecasting
def gm11(x0, n_predict):
    n = len(x0)
    x1 = np.cumsum(x0)  # Cumulative sum of the original data
    
    # Coefficient estimation
    B = np.array([-0.5 * (x1[1:] + x1[:-1]), np.ones(n-1)]).T
    Y = x0[1:]
    coeff = np.linalg.inv(B.T @ B) @ B.T @ Y
    
    a, b = coeff[0], coeff[1]
    
    # Predict the values
    x_hat = np.zeros(n + n_predict)
    x_hat[0] = x0[0]
    
    for k in range(1, n + n_predict):
        x_hat[k] = (x0[0] - b/a) * np.exp(-a * k) + b/a
    
    # Calculate the non-cumulative predicted values
    x0_hat = np.diff(x_hat)
    
    return x0_hat[:n], x0_hat[n:]  # First element is the fit, second is the forecast

# Function to compute RMAPE
def rmape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Load the data (assuming it's already loaded in DataFrame `df`)
data = {
    "Date": ["01-04-2018", "01-07-2018", "01-10-2018", "01-01-2019", "01-04-2019", "01-07-2019", "01-10-2019", "01-01-2020",
             "01-04-2020", "01-07-2020", "01-10-2020", "01-01-2021", "01-04-2021", "01-07-2021", "01-10-2021", "01-01-2022",
             "01-04-2022", "01-07-2022", "01-10-2022", "01-01-2023", "01-04-2023", "01-07-2023", "01-10-2023", "01-01-2024"],
    "Maharashtra": [17.2, 17.3, 18.7, 19.6, 19.0, 21.8, 22.1, 22.7, 13.7, 16.0, 22.2, 19.1, 18.3, 20.8, 22.2, 22.6, 23.0, 24.0, 24.0, 24.2, 24.3, 24.2, 24.9, 26.1],
    "Gujarat": [14.8, 16.2, 14.5, 15.0, 15.6, 16.2, 18.0, 18.4, 15.2, 17.8, 18.4, 19.2, 16.2, 14.8, 16.5, 15.7, 18.4, 20.6, 20.7, 23.8, 24.4, 24.5, 25.5, 25.2],
    "Madhya Pradesh": [15.5, 16.4, 15.4, 14.3, 13.2, 15.4, 16.9, 17.4, 12.6, 15.1, 15.7, 16.6, 14.5, 14.7, 15.8, 15.0, 15.6, 16.5, 16.5, 17.2, 15.4, 18.8, 22.1, 22.0],
    "Uttar Pradesh": [8.9, 8.8, 8.5, 7.1, 8.5, 8.8, 9.7, 10.7, 10.1, 10.1, 10.2, 9.7, 10.6, 9.5, 9.6, 9.1, 9.7, 9.8, 10.8, 11.9, 11.6, 13.0, 14.1, 13.5],
    "West Bengal": [20.9, 22.7, 21.8, 22.2, 22.2, 22.9, 23.6, 22.4, 19.0, 21.7, 22.2, 22.3, 20.0, 17.9, 19.8, 21.8, 22.4, 24.1, 25.6, 23.7, 23.9, 24.3, 25.4, 28.0],
    "Tamil Nadu": [22.0, 22.7, 23.1, 23.8, 23.6, 27.2, 27.0, 27.7, 23.3, 26.0, 26.3, 26.9, 22.9, 24.8, 23.2, 24.4, 24.9, 24.1, 25.6, 25.0, 24.3, 23.6, 23.7, 24.5],
    "India": [16.4, 17.1, 17.2, 16.9, 16.9, 18.3, 19.0, 19.6, 15.5, 17.1, 17.9, 18.7, 17.2, 17.6, 18.1, 18.3, 18.9, 19.7, 20.2, 20.6, 21.1, 21.9, 22.9, 23.4]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Initialize a dictionary to store forecasts and RMAPE values
forecasts = {}
rmapes = {}

for column in df.columns[1:]:  # Skip the 'Date' column
    x0 = df[column].values
    n_predict = 2  # Forecasting for the next 2 quarters
    
    # Apply the Grey Model
    fit, forecast = gm11(x0, n_predict)
    
    # If the forecast length is less than expected, fill the remaining with the last forecasted value
    if len(forecast) < n_predict:
        forecast = np.append(forecast, [forecast[-1]] * (n_predict - len(forecast)))
    
    # Calculate RMAPE for the model on the actual fitted data
    fitted_values, _ = gm11(x0[:-2], n_predict=2)  # Remove last two points and fit again
    rmape_value = rmape(x0[:-2], fitted_values[:len(x0)-2])
    
    # Store results
    forecasts[column] = forecast
    rmapes[column] = rmape_value

# Extending the dates for the forecast
dates_extended = pd.date_range(start=df['Date'].iloc[-1], periods=n_predict+1, freq='Q')[1:]

# Creating the result DataFrame
forecast_df = pd.DataFrame({'Date': dates_extended})
for column in forecasts:
    forecast_df[column] = forecasts[column]

print("Forecasts for the next two quarters:")
print(forecast_df)
print("\nRMAPE values for each state:")
print(rmapes)


# # ### **Comparison of Forecasted Values and RMSE (Male and Female)**
# 
# | **State**        | **Forecasted Values - Male** | **RMSE - Male**  | **Forecasted Values - Female** | **RMSE - Female** |
# |------------------|------------------------------|------------------|--------------------------------|-------------------|
# |                  | **2024-06-30** | **2024-09-30** |                                |                   |
# | **Maharashtra**  | 70.08104                     | 6.196502         | 26.096044                      | 2.171722          |
# | **Gujarat**      | 76.703585                    | 1.771820         | 25.453048                      | 1.927937          |
# | **Madhya Pradesh**| 71.48153                    | 3.368473         | 19.154863                      | 1.254401          |
# | **Uttar Pradesh**| 70.263078                    | 1.839155         | 13.196817                      | 0.829917          |
# | **West Bengal**  | 73.949496                    | 2.587878         | 24.807016                      | 1.686501          |
# | **Tamil Nadu**   | 68.199595                    | 2.121523         | 24.484645                      | 1.600463          |
# | **India**        | 69.452751                    | 2.432124         | 22.349133                      | 1.036997          |
# 
# ### Explanation
# 
# - **Forecasted Values - Male/Female**: The predicted values for male and female employment rates for the next two quarters.
# - **RMSE - Male/Female**: The RMSE (Root Mean Square Error) calculated for male and female data to evaluate the model's accuracy.
# 
# This table should reflect the forecasts and RMSE for both male and female categories using the GM(1,1) model.

# In[20]:


import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# Historical data for the last available quarter (01-01-2024)
historical_data = {
    'Maharashtra': 26.1,
    'Gujarat': 25.2,
    'Madhya Pradesh': 22.0,
    'Uttar Pradesh': 13.5,
    'West Bengal': 28.0,
    'Tamil Nadu': 24.5,
    'India': 23.4
}

# Forecasted data for 2024-Q2 (April to June)
forecasted_data = {
    'Maharashtra': 26.096044,
    'Gujarat': 25.453048,
    'Madhya Pradesh': 19.154863,
    'Uttar Pradesh': 13.196817,
    'West Bengal': 24.807016,
    'Tamil Nadu': 24.484645,
    'India': 22.349133
}

# Convert the data into pandas Series
historical_series = pd.Series(historical_data)
forecasted_series = pd.Series(forecasted_data)

# Conduct the paired t-test
t_stat, p_value = ttest_rel(historical_series, forecasted_series)

# Display the results
print("Paired t-test results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Determine if there is a significant difference
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the historical and forecasted data.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the historical and forecasted data.")


# # female workforce participation rate in rural and urban in state of maharashtra, MP and India.

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt

# Data provided
data = {
    'State': [
        'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
        'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
        'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh',
        'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh',
        'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India'
    ],
    'Year': [
        '01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19',
        '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22',
        '01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19',
        '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22',
        '01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19',
        '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22'
    ],
    'Area': [
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'
    ],
    'Employment Rate': [
        40.9, 21.1, 41.4, 22.9, 52.4, 26.9,
        48.7, 25.5, 50.0, 28.9, 55.1, 28.8,
        37.3, 20.7, 33.5, 17.9, 46.8, 24.1,
        51.4, 24.3, 50.3, 23.7, 55.6, 22.1,
        25.5, 19.8, 27.2, 20.2, 34.4, 23.3,
        38.4, 23.4, 38.4, 24.3, 43.4, 26.0
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Year' to datetime
df['Year'] = pd.to_datetime(df['Year'])

# Plotting
plt.figure(figsize=(14, 8))

# Plot Rural vs Urban for each State
for state in df['State'].unique():
    state_data = df[df['State'] == state]
    plt.plot(state_data[state_data['Area'] == 'Rural']['Year'], state_data[state_data['Area'] == 'Rural']['Employment Rate'], marker='o', label=f'{state} Rural')
    plt.plot(state_data[state_data['Area'] == 'Urban']['Year'], state_data[state_data['Area'] == 'Urban']['Employment Rate'], marker='o', label=f'{state} Urban')

plt.title('Employment Rate Comparison between Rural and Urban Areas')
plt.xlabel('Year')
plt.ylabel('Employment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()


# # applying linear regression model to forecast workforce for year 2024

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Data provided
data = {
    'State': [
        'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
        'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra', 'Maharashtra',
        'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh',
        'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Madhya Pradesh',
        'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India', 'India'
    ],
    'Year': [
        '01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19',
        '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22',
        '01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19',
        '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22',
        '01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19',
        '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22'
    ],
    'Area': [
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban',
        'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'
    ],
    'Employment Rate': [
        40.9, 21.1, 41.4, 22.9, 52.4, 26.9,
        48.7, 25.5, 50.0, 28.9, 55.1, 28.8,
        37.3, 20.7, 33.5, 17.9, 46.8, 24.1,
        51.4, 24.3, 50.3, 23.7, 55.6, 22.1,
        25.5, 19.8, 27.2, 20.2, 34.4, 23.3,
        38.4, 23.4, 38.4, 24.3, 43.4, 26.0
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Year' to datetime
df['Year'] = pd.to_datetime(df['Year'])

# Generate forecasted years
future_years = pd.date_range(df['Year'].max() + pd.DateOffset(months=12), periods=1, freq='A-JUL')

# Plotting
plt.figure(figsize=(14, 8))

# Plot Rural vs Urban for each State
for state in df['State'].unique():
    for area in ['Rural', 'Urban']:
        area_data = df[(df['State'] == state) & (df['Area'] == area)]
        
        # Linear Regression for forecasting
        X = np.array((area_data['Year'] - area_data['Year'].min()).dt.days).reshape(-1, 1)
        y = area_data['Employment Rate']
        
        model = LinearRegression().fit(X, y)
        
        # Forecasting next 1 year
        X_future = np.array((future_years - area_data['Year'].min()).days).reshape(-1, 1)
        y_future = model.predict(X_future)
        
        # Plot historical data
        plt.plot(area_data['Year'], area_data['Employment Rate'], marker='o', label=f'{state} {area}')
        
        # Plot forecast data with dotted lines, using the same color
        color = plt.gca().lines[-1].get_color()  # Get the color of the last line plotted
        plt.plot(future_years, y_future, 'o--', color=color, label=f'{state} {area} Forecast')

plt.title('Employment Rate Comparison between Rural and Urban Areas (Including Forecast)')
plt.xlabel('Year')
plt.ylabel('Employment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with your data
data = {
    'State and India': ['India'] * 12,
    'Year': ['01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19', '01-Jul-20', '01-Jul-20', '03-Jul-21', '04-Jul-21', '05-Jul-22', '06-Jul-22'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employment Rate': [25.5, 19.8, 27.2, 20.2, 34.4, 23.3, 38.4, 23.4, 38.4, 24.3, 43.4, 26.0]
}

df = pd.DataFrame(data)

# Convert 'Year' to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%y')

# Pivot the DataFrame to get 'Rural' and 'Urban' Employment Rates side by side
pivot_df = df.pivot(index='Year', columns='Area', values='Employment Rate')

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

pivot_df.plot(kind='bar', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Employment Rate (%)')
ax.set_title('Employment Rate Comparison: Rural vs Urban in India')
plt.xticks(rotation=45)
plt.legend(title='Area')
plt.tight_layout()
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the updated data
data = {
    'State and India': ['India'] * 12,
    'Year': ['01-Jul-17', '01-Jul-17', '01-Jul-18', '01-Jul-18', '01-Jul-19', '01-Jul-19', '01-Jul-20', '01-Jul-20', '01-Jul-21', '01-Jul-21', '01-Jul-22', '01-Jul-22'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employment Rate': [25.5, 19.8, 27.2, 20.2, 34.4, 23.3, 38.4, 23.4, 38.4, 24.3, 43.4, 26.0]
}

df = pd.DataFrame(data)

# Convert 'Year' to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%y')

# Pivot the DataFrame to get 'Rural' and 'Urban' Employment Rates side by side
pivot_df = df.pivot(index='Year', columns='Area', values='Employment Rate')

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

pivot_df.plot(kind='bar', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Employment Rate (%)')
ax.set_title('Female workforce participation rate Comparison: Rural vs Urban in India')
plt.xticks(rotation=45)
plt.legend(title='Area')
plt.tight_layout()
plt.show()

#current data

Maharashtra	2022-23	Rural	55.1
Maharashtra	2022-23	Urban	28.8
Madhya Pradesh 2022-23	Rural	55.6
Madhya Pradesh 2022-23	Urban	22.1
India 2022-23 Rural	43.4
India 2022-23 Urban	26




# State            Area    Forecast Year    Forecasted Employment Rate
0     Maharashtra  Rural    2023-24                   57.579435
1     Maharashtra  Urban    2023-24                  31.298839
2  Madhya Pradesh  Rural    2022-24                   60.761032
3  Madhya Pradesh  Urban    2023-24                   24.637221
4           India  Rural    2023-24                   47.515774
5           India  Urban    2023-24                   27.256833
# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the updated data
data = {
    'State and India': ['India'] * 12,
    'Year': ['2017-2018', '2017-2018', '2018-2019', '2018-2019', '2019-2020', '2019-2020', '2020-2021', '2020-2021', '2021-2022', '2021-2022', '2022-2023', '2022-2023'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employment Rate': [25.5, 19.8, 27.2, 20.2, 34.4, 23.3, 38.4, 23.4, 38.4, 24.3, 43.4, 26.0]
}

df = pd.DataFrame(data)

# Pivot the DataFrame to get 'Rural' and 'Urban' Employment Rates side by side
pivot_df = df.pivot(index='Year', columns='Area', values='Employment Rate')

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

pivot_df.plot(kind='bar', ax=ax, color=['purple', 'pink'])
ax.set_xlabel('Year')
ax.set_ylabel('Employment Rate (%)')
ax.set_title('Female workforce participation rate Comparison: Rural vs Urban in India')
plt.xticks(rotation=0)
plt.legend(title='Area')
plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with the updated data
data = {
    'State and India': ['India'] * 12,
    'Year': ['2017-2018', '2017-2018', '2018-2019', '2018-2019', '2019-2020', '2019-2020', '2020-2021', '2020-2021', '2021-2022', '2021-2022', '2022-2023', '2022-2023'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employment Rate': [23.7, 18.2, 25.5, 18.4, 32.2, 21.3, 28.6, 19, 27.9, 19.9, 33.2, 21.8]
}

df = pd.DataFrame(data)

# Pivot the DataFrame to get 'Rural' and 'Urban' Employment Rates side by side
pivot_df = df.pivot(index='Year', columns='Area', values='Employment Rate')

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

pivot_df.plot(kind='bar', ax=ax, color=['purple', 'pink'])
ax.set_xlabel('Year')
ax.set_ylabel('Employment Rate (%)')
ax.set_title('Female workforce participation rate Comparison: Rural vs Urban in India')
plt.xticks(rotation=0)
plt.legend(title='Area')
plt.tight_layout()
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from the table
data = {
    'Year': ['2017-2018', '2017-2018', '2018-2019', '2018-2019', '2019-2020', '2019-2020', '2020-2021', '2020-2021', '2021-2022', '2021-2022', '2022-2023', '2022-2023'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employer': [19, 23.7, 23.8, 22.3, 22.1, 21.5, 21.9, 26, 25.1, 26.7, 31, 25.9],
    'Helper in Household': [38.7, 11, 32.9, 9, 38.2, 9.8, 42.8, 12.4, 42.7, 12.7, 38.6, 11.9],
    'Regular Wage': [10.5, 52.1, 13.1, 57.3, 11.1, 56.7, 9.1, 50.1, 8.1, 50.3, 9.6, 53.5],
    'Casual Worker': [31.8, 13.1, 29, 9.3, 26.4, 8.5, 26.2, 11.5, 24.1, 10.3, 19.6, 7.6]
}

df = pd.DataFrame(data)

# Creating the bar graph
categories = ['Employer', 'Helper in Household', 'Regular Wage', 'Casual Worker']
years = df['Year'].unique()

fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.35
index = np.arange(len(years))

# Loop through each category and plot the bars for Rural and Urban separately
for i, category in enumerate(categories):
    ax.bar(index - bar_width/2 + i*bar_width/len(categories), df[df['Area'] == 'Rural'][category], bar_width/len(categories), label=f'Rural - {category}')
    ax.bar(index + bar_width/2 + i*bar_width/len(categories), df[df['Area'] == 'Urban'][category], bar_width/len(categories), label=f'Urban - {category}')

ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_title('Employment Types in Rural and Urban Areas Over Years')
ax.set_xticks(index)
ax.set_xticklabels(years, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from the table
data = {
    'Year': ['2017-2018', '2017-2018', '2018-2019', '2018-2019', '2019-2020', '2019-2020', '2020-2021', '2020-2021', '2021-2022', '2021-2022', '2022-2023', '2022-2023'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employer': [19, 23.7, 23.8, 22.3, 22.1, 21.5, 21.9, 26, 25.1, 26.7, 31, 25.9],
    'Helper in Household': [38.7, 11, 32.9, 9, 38.2, 9.8, 42.8, 12.4, 42.7, 12.7, 38.6, 11.9],
    'Regular Wage': [10.5, 52.1, 13.1, 57.3, 11.1, 56.7, 9.1, 50.1, 8.1, 50.3, 9.6, 53.5],
    'Casual Worker': [31.8, 13.1, 29, 9.3, 26.4, 8.5, 26.2, 11.5, 24.1, 10.3, 19.6, 7.6]
}

df = pd.DataFrame(data)

# Creating the bar graph for Rural areas
categories = ['Employer', 'Helper in Household', 'Regular Wage', 'Casual Worker']
years = df['Year'].unique()

fig, ax = plt.subplots(2, 1, figsize=(12, 16))

# Plot for Rural area
rural_df = df[df['Area'] == 'Rural']
index = np.arange(len(years))

for i, category in enumerate(categories):
    ax[0].bar(index + i * 0.2, rural_df[category], width=0.2, label=category)

ax[0].set_xlabel('Year')
ax[0].set_ylabel('Percentage')
ax[0].set_title('Employment Types in Rural Areas Over Years')
ax[0].set_xticks(index + 0.3)
ax[0].set_xticklabels(years, rotation=45)
ax[0].legend()

# Plot for Urban area
urban_df = df[df['Area'] == 'Urban']

for i, category in enumerate(categories):
    ax[1].bar(index + i * 0.2, urban_df[category], width=0.2, label=category)

ax[1].set_xlabel('Year')
ax[1].set_ylabel('Percentage')
ax[1].set_title('Employment Types in Urban Areas Over Years')
ax[1].set_xticks(index + 0.3)
ax[1].set_xticklabels(years, rotation=45)
ax[1].legend()

plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    'Year': ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023'],
    'Employer_Rural': [19, 23.8, 22.1, 21.9, 25.1, 31],
    'Helper in Household_Rural': [38.7, 32.9, 38.2, 42.8, 42.7, 38.6],
    'Regular Wage_Rural': [10.5, 13.1, 11.1, 9.1, 8.1, 9.6],
    'Casual Worker_Rural': [31.8, 29, 26.4, 26.2, 24.1, 19.6],
    'Employer_Urban': [23.7, 22.3, 21.5, 26, 26.7, 25.9],
    'Helper in Household_Urban': [11, 9, 9.8, 12.4, 12.7, 11.9],
    'Regular Wage_Urban': [52.1, 57.3, 56.7, 50.1, 50.3, 53.5],
    'Casual Worker_Urban': [13.1, 9.3, 8.5, 11.5, 10.3, 7.6]
}

df = pd.DataFrame(data)

# Plotting for Rural Areas
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(df['Year']))

ax.bar(index, df['Employer_Rural'], bar_width, label='Employer', color='pink')
ax.bar(index + bar_width, df['Helper in Household_Rural'], bar_width, label='Helper in Household', color='purple')
ax.bar(index + 2*bar_width, df['Regular Wage_Rural'], bar_width, label='Regular Wage', color='skyblue')
ax.bar(index + 3*bar_width, df['Casual Worker_Rural'], bar_width, label='Casual Worker', color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_title('Employment Distribution in Rural Areas (2017-2023)')
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(df['Year'], rotation=45)
ax.legend()

plt.tight_layout()
plt.ylim(0,50)
plt.show()

# Plotting for Urban Areas
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(df['Year']))

ax.bar(index, df['Employer_Urban'], bar_width, label='Employer', color='pink')
ax.bar(index + bar_width, df['Helper in Household_Urban'], bar_width, label='Helper in Household', color='purple')
ax.bar(index + 2*bar_width, df['Regular Wage_Urban'], bar_width, label='Regular Wage', color='skyblue')
ax.bar(index + 3*bar_width, df['Casual Worker_Urban'], bar_width, label='Casual Worker', color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Percentage')
ax.set_title('Employment Distribution in Urban Areas (2017-2023)')
ax.set_xticks(index + 1.5*bar_width)
ax.set_xticklabels(df['Year'], rotation=45)
ax.legend()

plt.tight_layout()
plt.ylim(0,60)
plt.show()


# In[1]:


import numpy as np
import pandas as pd

# Extracted data
data = {
    'Year': ['2017-2018', '2017-2018', '2018-2019', '2018-2019', '2019-2020', '2019-2020', 
             '2020-2021', '2020-2021', '2021-2022', '2021-2022', '2022-2023', '2022-2023'],
    'Area': ['Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban', 
             'Rural', 'Urban', 'Rural', 'Urban', 'Rural', 'Urban'],
    'Employment Rate': [25.5, 19.8, 27.2, 20.2, 34.4, 23.3, 38.4, 23.4, 38.4, 24.3, 43.4, 26.0]
}

# Create DataFrame
df = pd.DataFrame(data)


# In[1]:


def grey_model(x0):
    n = len(x0)
    x1 = np.cumsum(x0)  # Accumulated generation series
    B = np.array([-0.5 * (x1[i] + x1[i-1]) for i in range(1, n)])
    Y = np.array(x0[1:])
    B = B.reshape(-1, 1)
    a, u = np.linalg.inv(B.T @ B) @ B.T @ Y  # Least squares method to solve
    a, u = -a, u
    x1_hat = [(x0[0] - u/a) * np.exp(a * i) + u/a for i in range(n)]
    x0_hat = [x1_hat[0]] + [x1_hat[i] - x1_hat[i-1] for i in range(1, n)]
    return np.array(x0_hat), (x0[0] - u/a) * np.exp(a * (n)) + u/a


# In[13]:


import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Data for Rural and Urban areas
rural_data = [25.5, 27.2, 34.4, 38.4, 38.4, 43.4]  # Data from 2017-2018 to 2022-2023
urban_data = [19.8, 20.2, 23.3, 23.4, 24.3, 26]    # Data from 2017-2018 to 2022-2023

# Convert to Pandas Series
rural_series = pd.Series(rural_data)
urban_series = pd.Series(urban_data)

# Fit the ETS model
rural_model = ExponentialSmoothing(rural_series, trend='add', seasonal=None).fit()
urban_model = ExponentialSmoothing(urban_series, trend='add', seasonal=None).fit()

# Forecast the next period (2023-2024)
rural_forecast = rural_model.forecast(1)
urban_forecast = urban_model.forecast(1)

# Print the forecasted values
print(f"Forecasted Employment Rate for Rural Area for 2023-2024: {rural_forecast.values[0]}")
print(f"Forecasted Employment Rate for Urban Area for 2023-2024: {urban_forecast.values[0]}")


# In[14]:


from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
rural_arima_model = ARIMA(rural_series, order=(1, 1, 1)).fit()
urban_arima_model = ARIMA(urban_series, order=(1, 1, 1)).fit()

# Forecast the next period (2023-2024)
rural_arima_forecast = rural_arima_model.forecast(1)
urban_arima_forecast = urban_arima_model.forecast(1)

# Print the forecasted values
print(f"Forecasted Employment Rate for Rural Area for 2023-2024: {rural_arima_forecast.values[0]}")
print(f"Forecasted Employment Rate for Urban Area for 2023-2024: {urban_arima_forecast.values[0]}")
from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming `actual_values` is a list/array of actual values
# and `predicted_values` is a list/array of predicted values

def calculate_rmse(actual_values, predicted_values):
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse
rmse = calculate_rmse(actual_values, predicted_values)
print(f"RMSE: {rmse}")


# In[11]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming `actual_values` is a list/array of actual values
# and `predicted_values` is a list/array of predicted values

def calculate_rmse(actual_values, predicted_values):
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    return rmse

# Example usage (replace with your actual and predicted data)
actual_values = [25.5, 27.2, 34.4, 38.4, 38.4, 43.4]
predicted_values = [26.0, 27.5, 33.9, 38.2, 38.5, 43.0]

rmse = calculate_rmse(actual_values, predicted_values)
print(f"RMSE: {rmse}")


# In[12]:


# Calculate RMSE for ETS model
rural_ets_pred = rural_model.fittedvalues  # Fitted values by ETS model
urban_ets_pred = urban_model.fittedvalues

rural_ets_rmse = calculate_rmse(rural_series, rural_ets_pred)
urban_ets_rmse = calculate_rmse(urban_series, urban_ets_pred)

# Calculate RMSE for ARIMA model
rural_arima_pred = rural_arima_model.fittedvalues  # Fitted values by ARIMA model
urban_arima_pred = urban_arima_model.fittedvalues

rural_arima_rmse = calculate_rmse(rural_series, rural_arima_pred)
urban_arima_rmse = calculate_rmse(urban_series, urban_arima_pred)

# Compare RMSE values
print(f"ETS RMSE (Rural): {rural_ets_rmse}")
print(f"ETS RMSE (Urban): {urban_ets_rmse}")

print(f"ARIMA RMSE (Rural): {rural_arima_rmse}")
print(f"ARIMA RMSE (Urban): {urban_arima_rmse}")


# In[17]:


import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Data for Rural and Urban areas
rural_data = [25.5, 27.2, 34.4, 38.4, 38.4, 43.4]  # Data from 2017-2018 to 2022-2023
urban_data = [19.8, 20.2, 23.3, 23.4, 24.3, 26]    # Data from 2017-2018 to 2022-2023

# Convert to Pandas Series
rural_series = pd.Series(rural_data)
urban_series = pd.Series(urban_data)

# Split data into training and test sets (last period as test set)
rural_train, rural_test = rural_series[:-1], rural_series[-1:]
urban_train, urban_test = urban_series[:-1], urban_series[-1:]

# Fit the ETS model on training data
rural_model = ExponentialSmoothing(rural_train, trend='add', seasonal=None).fit()
urban_model = ExponentialSmoothing(urban_train, trend='add', seasonal=None).fit()

# Forecast the next period (2023-2024)
rural_forecast = rural_model.forecast(1)
urban_forecast = urban_model.forecast(1)

# Calculate RMSE on test set
rural_rmse = np.sqrt(mean_squared_error(rural_test, rural_forecast))
urban_rmse = np.sqrt(mean_squared_error(urban_test, urban_forecast))

# Print the forecasted values and RMSE
print(f"Forecasted Employment Rate for Rural Area for 2023-2024: {rural_forecast.values[0]}")
print(f"Forecasted Employment Rate for Urban Area for 2023-2024: {urban_forecast.values[0]}")
print(f"RMSE for Rural Area Forecast: {rural_rmse}")
print(f"RMSE for Urban Area Forecast: {urban_rmse}")


# In[1]:


import matplotlib.pyplot as plt

# Historical data (2017-2018 to 2022-2023)
years = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
rural_data = [25.5, 27.2, 34.4, 38.4, 38.4, 43.4]
urban_data = [19.8, 20.2, 23.3, 23.4, 24.3, 26]

# Add the forecasted values for 2023-2024
years.append('2023-2024')
rural_data.append(43.88003684679764)
urban_data.append(25.859999877617266)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(years[:-1], rural_data[:-1], marker='o', label='Rural Area - Historical')
plt.plot(years[:-1], urban_data[:-1], marker='o', label='Urban Area - Historical')
plt.plot(years[-2:], rural_data[-2:], 'r--', marker='o', label='Rural Area - Forecasted')
plt.plot(years[-2:], urban_data[-2:], 'b--', marker='o', label='Urban Area - Forecasted')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Employment Rate')
plt.title('Historical and Forecasted Employment Rates for Rural and Urban Areas')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# In[10]:


import matplotlib.pyplot as plt

# Historical data (2017-2018 to 2022-2023)
years = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023']
rural_data = [23.7, 25.5, 32.2, 28.6, 27.9, 33.2]
urban_data = [18.2, 18.4, 21.3, 19, 19.9, 21.8]

# Add the forecasted values for 2023-2024
years.append('2023-2024')
rural_data.append(31.03)  # Replace with your forecasted value
urban_data.append(20.56)  # Replace with your forecasted value

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(years[:-1], rural_data[:-1], marker='o', label='Rural Area - Historical', color='blue')
plt.plot(years[:-1], urban_data[:-1], marker='o', label='Urban Area - Historical', color='purple')
plt.plot(years[-2:], rural_data[-2:], '--', marker='o', label='Rural Area - Forecasted', color='blue')
plt.plot(years[-2:], urban_data[-2:], '--', marker='o', label='Urban Area - Forecasted', color='purple')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Employment Rate (%)')
plt.title('Historical and Forecasted Employment Rates for Rural and Urban Areas')
plt.legend()

# Setting y-axis ticks
plt.yticks(range(0, 41, 5))

# Show the plot
plt.grid(True)
plt.show()


# In[7]:


import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Updated data for Rural and Urban areas
rural_data = [23.7, 25.5, 32.2, 28.6, 27.9, 33.2]  # Data from 2017-2018 to 2022-2023
urban_data = [18.2, 18.4, 21.3, 19, 19.9, 21.8]    # Data from 2017-2018 to 2022-2023

# Convert to Pandas Series
rural_series = pd.Series(rural_data)
urban_series = pd.Series(urban_data)

# Split data into training and test sets (last period as test set)
rural_train, rural_test = rural_series[:-1], rural_series[-1:]
urban_train, urban_test = urban_series[:-1], urban_series[-1:]

# Fit the ETS model on training data
rural_model = ExponentialSmoothing(rural_train, trend='add', seasonal=None).fit()
urban_model = ExponentialSmoothing(urban_train, trend='add', seasonal=None).fit()

# Forecast the next period (2023-2024)
rural_forecast = rural_model.forecast(1)
urban_forecast = urban_model.forecast(1)

# Calculate RMSE on test set
rural_rmse = np.sqrt(mean_squared_error(rural_test, rural_forecast))
urban_rmse = np.sqrt(mean_squared_error(urban_test, urban_forecast))

# Print the forecasted values and RMSE
print(f"Forecasted Employment Rate for Rural Area for 2023-2024: {rural_forecast.values[0]}")
print(f"Forecasted Employment Rate for Urban Area for 2023-2024: {urban_forecast.values[0]}")
print(f"RMSE for Rural Area Forecast: {rural_rmse}")
print(f"RMSE for Urban Area Forecast: {urban_rmse}")


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Historical data (2017-2018 to 2022-2023)
years = np.array([2017, 2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
rural_data = np.array([23.7, 25.5, 32.2, 28.6, 27.9, 33.2])
urban_data = np.array([18.2, 18.4, 21.3, 19.0, 19.9, 21.8])

# Fit linear regression models for rural and urban data
rural_model = LinearRegression().fit(years, rural_data)
urban_model = LinearRegression().fit(years, urban_data)

# Forecast for 2023-2024 (year = 2023)
forecast_year = np.array([[2023]])
rural_forecast = rural_model.predict(forecast_year)[0]
urban_forecast = urban_model.predict(forecast_year)[0]

print(f"Forecasted Rural Employment Rate (2023-2024): {rural_forecast:.2f}")
print(f"Forecasted Urban Employment Rate (2023-2024): {urban_forecast:.2f}")

# Extend the data with forecasted values
years = np.append(years, 2023).reshape(-1)
rural_data = np.append(rural_data, rural_forecast)
urban_data = np.append(urban_data, urban_forecast)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(years[:-1], rural_data[:-1], marker='o', label='Rural Area - Historical', color='blue')
plt.plot(years[:-1], urban_data[:-1], marker='o', label='Urban Area - Historical', color='purple')
plt.plot(years[-2:], rural_data[-2:], '--', marker='o', label='Rural Area - Forecasted', color='blue')
plt.plot(years[-2:], urban_data[-2:], '--', marker='o', label='Urban Area - Forecasted', color='purple')

# Adding labels, title, and grid
plt.xlabel('Year')
plt.ylabel('Employment Rate (%)')
plt.title('Historical and Forecasted Employment Rates for Rural and Urban Areas')
plt.legend(loc='upper left')
plt.grid(True)
plt.yticks(range(0, 41, 5))

# Show the plot
plt.show()

