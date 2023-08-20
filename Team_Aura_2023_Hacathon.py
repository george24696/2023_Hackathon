#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import glob
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
import folium
from folium.plugins import HeatMap
from mpl_toolkits.mplot3d import Axes3D


# In[28]:


# Get a list of all the .csv files
files = glob.glob('*.csv')

# Create an empty list to store the data
data = []

# Loop through the files and read the data
for file in files:
    df = pd.read_csv(file)
    data.append(df)

# Concatenate the data into one big DataFrame
data = pd.concat(data, ignore_index=True)


# In[29]:


# Convert the timestamp column to a datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])


# ### Cleaning Data

# In[30]:


# Replace missing acceleration values with zero
data['x_accel'] = data['x_accel'].fillna(0)
data['y_accel'] = data['y_accel'].fillna(0)
data['z_accel'] = data['z_accel'].fillna(0)


# In[5]:


print(data.info())


# In[31]:


# Count the number of null values in each column
null_counts = data.isnull().sum()
print(null_counts)


# In[32]:


# Drop the missing values in latitude and longitude rows
data = data.dropna(subset=['coordinate_latitude', 'coordinate_longitude'])

# For the odometer, remove the missing rows
data = data.dropna(subset=['odometer'])

# Remove the whole altitude column
data = data.drop('altitude', axis=1)

# Remove SP_CODE rows with null values
data = data.dropna(subset=['SP_CODE'])


# In[33]:


# Use interpolation to fill in missing road_speed values
data['road_speed'] = data['road_speed'].interpolate()


# In[34]:


# Count the number of null values in each column
null_counts = data.isnull().sum()
print(null_counts)


# In[35]:


# Remove remaining road_speed rows with null values
data = data.dropna(subset=['road_speed'])


# In[36]:


# Count the number of null values in each column
null_counts = data.isnull().sum()
print(null_counts)


# ### Data is clean

# Visuals 

# In[21]:


# Extract the day of the week and hour of the day from the timestamp column
data['day_of_week'] = data['timestamp'].dt.day_name()
data['hour_of_day'] = data['timestamp'].dt.hour

# Count the number of taxis in operation at each hour of the day for each day of the week
taxi_count = data.groupby(['day_of_week', 'hour_of_day'])['vehicleid'].count().unstack()

# Reorder the index of taxi_count so that the days of the week start from Monday to Sunday
taxi_count = taxi_count.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create a heatmap of taxi activity by day of week and hour of day
sns.heatmap(taxi_count, cmap='YlGnBu')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.title('Heatmap of Taxi Activity by Day of Week and Hour of Day')
plt.show()


# In[23]:


def cluster_taxis(speeding_percentage):
    # Define the cluster labels
    cluster_labels = ['Low Risk Speeding', 'Medium Risk Speeding', 'High Risk Speeding']

    # Cluster the taxis based on their speeding percentage
    clusters = pd.cut(speeding_percentage, bins=[0, 5, 10, 100], labels=cluster_labels, right=False)

    # Create a DataFrame to store the results
    df = pd.DataFrame({'vehicleid': clusters.index, 'speeding_cluster': clusters.values})

    return df

# Cluster the taxis based on their speeding percentage
clusters = cluster_taxis(speeding_percentage)

# Display the resulting table
print(clusters)


# In[26]:


import matplotlib.pyplot as plt

# Cluster the taxis based on their speeding percentage
clusters = cluster_taxis(speeding_percentage)

# Count the number of taxis in each cluster
cluster_counts = clusters['speeding_cluster'].value_counts()

# Create a bar chart to display the number of taxis in each cluster
cluster_counts.plot(kind='bar')
plt.xlabel('Speeding Clusters')
plt.ylabel('Number of Taxis')
plt.title('Speeding Taxi Clusters')
plt.show()


# ### Data has been re-imported and recleaned as "data['speeding'] = data['speed'] > data['road_speed']" removes values in the Database
# 

# In[38]:


# Create a scatter plot of acceleration vs. speed
sns.scatterplot(x='speed', y='linear_g', data=data)
plt.xlabel('Speed')
plt.ylabel('Acceleration')
plt.title('Acceleration(linear_g) vs. Speed')
plt.show()


# In[43]:


def cluster_taxis_acceleration(data):
    # Cluster the taxis based on their linear_g values
    clusters = pd.cut(data['linear_g'], bins=[-float('inf'), -0.1, 0.1, float('inf')], labels=['Aggressive deceleration', 'Good acceleration', 'Aggressive acceleration'], right=True)

    # Create a DataFrame to store the results
    df = pd.DataFrame({'vehicleid': data['vehicleid'], 'acceleration_cluster': clusters})

    return df

# Cluster the taxis based on their acceleration behavior
clusters = cluster_taxis_acceleration(data)

# Count the number of taxis in each cluster
cluster_counts = clusters['acceleration_cluster'].value_counts()


# Display the resulting table
print(clusters)

# Create a bar chart to display the number of taxis in each cluster
cluster_counts.plot(kind='bar')
plt.xlabel('Acceleration Clusters')
plt.ylabel('Number of Taxis')
plt.title('Taxi Acceleration Clusters')
plt.show()


# In[44]:


# Cluster the taxis based on their acceleration behavior
clusters = cluster_taxis_acceleration(data)

# Count the number of taxis in each cluster
cluster_counts = clusters['acceleration_cluster'].value_counts()

# Calculate the percentage of taxis in each cluster
cluster_percentages = cluster_counts / cluster_counts.sum() * 100

# Create a table to display the acceleration percentages
table_data = [['Aggressive deceleration', cluster_percentages['Aggressive deceleration']],
              ['Good acceleration', cluster_percentages['Good acceleration']],
              ['Aggressive acceleration', cluster_percentages['Aggressive acceleration']]]
table = pd.DataFrame(table_data, columns=['Acceleration', 'Percentage'])
display(table)


# 
# 

# In[45]:


# Cluster the taxis based on their acceleration behavior
clusters = cluster_taxis_acceleration(data)

# Create a table to display the vehicleids for each cluster
table = clusters.pivot_table(index='vehicleid', columns='acceleration_cluster', aggfunc=len, fill_value=0)
display(table)


# In[22]:


# Create a new column that indicates whether the speed is greater than the road speed
data['speeding'] = data['speed'] > data['road_speed']

# Group the data by vehicleid and calculate the percentage of time each taxi is speeding
speeding_percentage = data.groupby('vehicleid')['speeding'].mean() * 100

# Sort the speeding percentages in ascending order
speeding_percentage = speeding_percentage.sort_values(ascending=True)

# Create a horizontal bar plot to display the speeding percentage for each taxi
speeding_percentage.plot(kind='barh', figsize=(10, 30))
plt.xlabel('Speeding Percentage (%)')
plt.ylabel('Vehicle ID')
plt.title('Taxi Speeding Percentage')
plt.show()


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt

# Calculate the net g-force for each row
data['net_g_force'] = (data['x_accel']**2 + data['y_accel']**2 + data['z_accel']**2)**0.5 / 9.81

# Group the data by vehicleid and calculate the average net g-force for each taxi
avg_net_g_force = data.groupby('vehicleid')['net_g_force'].mean()

# Create a bar chart to display the average net g-force for each taxi
avg_net_g_force.plot(kind='bar', figsize=(10, 5))
plt.xlabel('Vehicle ID')
plt.ylabel('Average Net G-Force')
plt.title('Taxi Average Net G-Force')
plt.show()


# In[47]:


import pandas as pd
import matplotlib.pyplot as plt

# Calculate the net g-force for each row
data['net_g_force'] = (data['x_accel']**2 + data['y_accel']**2 + data['z_accel']**2)**0.5 / 9.81

# Group the data by vehicleid and calculate the average net g-force for each taxi
avg_net_g_force = data.groupby('vehicleid')['net_g_force'].mean()

# Sort the average net g-force in ascending order
avg_net_g_force = avg_net_g_force.sort_values(ascending=True)

# Create a horizontal bar plot to display the average net g-force for each taxi
avg_net_g_force.plot(kind='barh', figsize=(10, 30))
plt.xlabel('Average Net G-Force')
plt.ylabel('Vehicle ID')
plt.title('Taxi Average Net G-Force')
plt.show()


# ## These taxis have the highest avarage net g force

# In[50]:


def display_top_5(data):
    # Calculate the net g-force for each row
    data['net_g_force'] = (data['x_accel']**2 + data['y_accel']**2 + data['z_accel']**2)**0.5 / 9.81

    # Group the data by vehicleid and calculate the average net g-force for each taxi
    avg_net_g_force = data.groupby('vehicleid')['net_g_force'].mean()

    # Sort the average net g-force in descending order and select the top 5
    top_5 = avg_net_g_force.sort_values(ascending=False).head(5)

    # Display the top 5 vehicleid in a table
    print(top_5.to_frame())
    
    
display_top_5(data)


# ### Below are the operationg hours of all the taxis

# In[14]:



# Extract the date and time from the timestamp column
data['date'] = data['timestamp'].dt.date
data['time'] = data['timestamp'].dt.time

# Convert the time values to seconds since midnight
data['time_seconds'] = data['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

# Filter the data to only include rows where the speed is greater than zero
data = data[data['speed'] > 0]

# Group the data by vehicleid
grouped_data = data.groupby('vehicleid')

# Loop through the grouped data and create a scatter plot for each taxi
for vehicleid, taxi_data in grouped_data:
    fig, ax = plt.subplots()
    ax.scatter(taxi_data['time_seconds'], taxi_data['date'])

    # Format the x-axis to show hours of the day
    ax.xaxis.set_major_locator(plt.MultipleLocator(3600))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x // 3600):02d}:00'))

    # Set the limits of the x-axis to remove the "-01:00" and "25:00" labels
    ax.set_xlim(0, 24 * 3600)

    # Rotate the x-axis labels for readability
    plt.xticks(rotation=90)
    
    # Print out the vehicleid
    print(f'vehicleid: {data["vehicleid"].iloc[0]}')

    # Show the plot
    plt.show()

