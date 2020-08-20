import numpy as np # linear algebra
import pandas as pd 

#In Jupyter directory, created folder named - nyc311project and can see content of this folder as follows
import os

#Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns 
#%matplotlib inline

# Question 1.) Import a 311 NYC service request.

# Solution 1

# Read csv
df_nyc = pd.read_csv("311_Service_Requests_from_2010_to_Present.csv")

# We are keeping one orig copy also with us
df_orig = pd.read_csv("311_Service_Requests_from_2010_to_Present.csv")

# To Read Top 5 records
print(df_nyc.head())

# Check shape of DataFrame
df_nyc.shape

# See columns
df_nyc.columns

# First we should check which column has how many missing values
print(df_nyc.isnull().sum())

# As we seen Closed Date is important column and have many missing values
print(df_nyc[df_nyc['Closed Date'].isnull()])

# For our future exploration on Closed Date column we have noted down one row by its unique key column to check changes everytime we do something for Closed Date or related column
print(df_nyc[df_nyc['Unique Key'] == 32305700])

# We check data type of each column
df_nyc.dtypes

# Question 2.) Read or convert the columns ‘Created Date’ and Closed Date’ to datetime datatype and create a new column ‘Request_Closing_Time’ as the time elapsed between request creation and request closing. (Hint: Explore the package/module datetime)

# Solution 2
import datetime as dt
import time, datetime

# Convert "Closed Date" to datetime dtype
df_nyc['Closed Date'] = pd.to_datetime(df_nyc['Closed Date'])
print(df_nyc['Closed Date'].dtype)


# Convert "Created Date" to datetime dtype
df_nyc['Created Date'] = pd.to_datetime(df_nyc['Created Date'])
print(df_nyc['Created Date'].dtype)


# Create new column Request_Closing_Time with time taken to close complain
df_nyc['Request_Closing_Time'] = df_nyc['Closed Date'] - df_nyc['Created Date']

print(df_nyc['Request_Closing_Time'].head())


# Question 3.: Provide major insights/patterns that you can offer in a visual format (graphs or tables); at least 4 major conclusions that you can come up with after generic data mining.

# Solution 3
# From here starting Insight
# Insight - 1 - Categorize Request_Closing_Time as follows -
# Below 2 hours - Fast, Between 2 to 4 hours - Acceptable, Between 4 to 6 - Slow, More than 6 hours - Very Slow
# For this, first will create new column Request_Closing_In_Hr and then create new column - Request_Closing_Time_Category

# Function to convert TimeDelta in Hour
def toHr(timeDel):
    days = timeDel.days
    hours = round(timeDel.seconds/3600, 2)
    result = (days * 24) + hours
    #print(days)
    #print(hours)
    return result
    #return round(pd.Timedelta(timeDel).seconds / 3600, 2)

# Testing of function with days
test_days = df_nyc[df_nyc['Unique Key'] == 32122264]['Request_Closing_Time']
print(toHr(test_days[27704]))
print(test_days[27704])
print(test_days.dtype)


# Apply this function to every row of column Request_Closing_Time
df_nyc['Request_Closing_In_Hr'] = df_nyc['Request_Closing_Time'].apply(toHr)

print(df_nyc['Request_Closing_In_Hr'].head())


import math

# Function to categorize hours - Less than 2 hours - Fast, Between 2 to 4 hours - Acceptable, Between 4 to 6 - Slow, More than 6 hours - Very Slow
def hrToCategory(hr):
    if (math.isnan(hr)):
        return 'Unspecified'
    elif (hr < 2.0):
        return 'Fast'
    elif (4.0 > hr >= 2.0):
        return 'Acceptable'
    elif (6.0 > hr >= 4.0):
        return 'Slow'
    else:
        return 'Very Slow'

# Testing function
print(hrToCategory(1.99))

# Create new column Request_Closing_Time_Category and apply function on column Request_Closing_In_Hr

df_nyc['Request_Closing_Time_Category'] = df_nyc['Request_Closing_In_Hr'].apply(hrToCategory)

df_nyc['Request_Closing_Time_Category'].head()

print(df_nyc['Request_Closing_Time_Category'].value_counts())

# Create Bar plot for Request_Closing_Time_Category to check frequency in Request_Closing_Time_Category and it prove Most count is in Fast category means closed less than 2 hours
df_nyc['Request_Closing_Time_Category'].value_counts().plot(kind="barh", color=list('rgbkymc'), alpha=0.7, figsize=(15,3))
plt.show()


print(df_nyc.head())

# Insight 2 - To check with Month have Complain creation most and least

# We will create one column with Create_Month name

# Created Series for months in text format
monthSeries = pd.Series({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
print(monthSeries)
print(monthSeries[12])


print(df_nyc['Created Date'].dtype)

# Function to fetch month from Created Date column

def getMonth(cDate):
    a = str(cDate)
    datee = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
    return monthSeries[datee.month]

# Test function getMonth
print(df_nyc['Created Date'][0])
print(getMonth(df_nyc['Created Date'][0]))


# Created new column Created_Month and kept all text format months in that column

df_nyc['Created_Month'] = df_nyc['Created Date'].apply(getMonth)
print(df_nyc['Created_Month'])


df_nyc.head()

print(df_nyc['Created_Month'].value_counts())

# Create Bar plot for Complain Created Month to check frequency and it prove Most count is in May month and least is in March and in January there is no any complain
df_nyc['Created_Month'].value_counts().plot(kind="barh", color=list('rgbkymc'), alpha=0.7, figsize=(15,3))
plt.show()


# To confirm doubt of January doesn't have any value, we used original dataframe and check if any entry for Jan month
print(df_orig[df_orig['Created Date'].str.startswith('01/')])


# Insight - 3
# Check count in each complain type - sorted decreasing order
print(df_nyc['Complaint Type'].value_counts())


# Create Bar plot for complain type to check frequency in Complain Type
df_nyc['Complaint Type'].value_counts().plot(kind="barh", color=list('rgbkymc'), alpha=0.7, figsize=(15,10))
plt.show()

# Insight 4
# Let's check count for status type
print(df_nyc['Status'].value_counts())

# Draw Bar lot for Status
from matplotlib import style
style.use('ggplot')
df_nyc['Status'].value_counts().plot(kind='bar', color=list('rgbkymc'))
plt.show()


# Question 4.: Order the complaint types based on the average ‘Request_Closing_Time’, grouping them for different locations.

# Solution 4:

# For location we can choose here City, so first check if there is missing values there
print(df_nyc['City'].isnull().sum())


# Fill all missing value with some default value here i used - Not Available
df_nyc['City'].fillna('Not Available', inplace=True)

print(df_nyc['City'].head())


print(df_nyc['City'])

# Group them for City (location) first and Complain Type in that
df_nyc_grouped = df_nyc.groupby(['City', 'Complaint Type'])

# get average of this grouped dataframe, and get Request_Closing_Time column from there
df_nyc_mean = df_nyc_grouped.mean()['Request_Closing_In_Hr']
print(df_nyc_mean.isnull().sum())


# Group by City(location) first and then Complain Type and showing average of Request Closing in Hour
df_nyc_grouped = df_nyc.groupby(['City','Complaint Type']).agg({'Request_Closing_In_Hr': 'mean'})
print(df_nyc_grouped)


# Check if any value is NaN
print(df_nyc_grouped[df_nyc_grouped['Request_Closing_In_Hr'].isnull()])


# Check total rows
print(df_nyc_grouped)
  

# drop null values from this group
df_nyc_grouped_withoutna = df_nyc_grouped.dropna()

# verify if new group has null values
print(df_nyc_grouped_withoutna.isnull().sum())


# verify number of rows after dropping null values
print(df_nyc_grouped_withoutna)


# Sorting by column - Request_Closing_In_Hr for City on grouped
df_nyc_sorted = df_nyc_grouped_withoutna.sort_values(['City', 'Request_Closing_In_Hr'])
print(df_nyc_sorted)


# Question 5: Perform a statistical test for the following:
# Please note: For the below statements you need to state the Null and Alternate and then provide a statistical test to accept or reject the Null Hypothesis along with the corresponding ‘p-value’.

# Whether the average response time across complaint types is similar or not (overall)
# Are the type of complaint or service requested and location related?

import scipy.stats as stats
from math import sqrt

##### Try ANOVA for first one

# H0 : All Complain Types average response time mean is similar
# H1 : Not similar

print(df_nyc['Complaint Type'].value_counts())


top5_complaints_type = df_nyc['Complaint Type'].value_counts()[:5]
print(top5_complaints_type)


top5_complaints_type_names = top5_complaints_type.index
print(top5_complaints_type_names)


sample_data = df_nyc.loc[df_nyc['Complaint Type'].isin(top5_complaints_type_names), ['Complaint Type', 'Request_Closing_In_Hr']]
print(sample_data.head())


print(sample_data.shape)


print(sample_data.isnull().sum())
#sample_data[~sample_data.isin(['NaN', 'NaT']).any(axis=1)]
#sample_data[sample_data.isnull()]

sample_data.dropna(how='any', inplace=True)
print(sample_data.isnull().sum())
# sample_data_without_null[sample_data_without_null.isnull()]


sample_data.shape


s1 = sample_data[sample_data['Complaint Type'] == top5_complaints_type_names[0]].Request_Closing_In_Hr
s1.head()


s2 = sample_data[sample_data['Complaint Type'] == top5_complaints_type_names[1]].Request_Closing_In_Hr
s2.head()


s3 = sample_data[sample_data['Complaint Type'] == top5_complaints_type_names[2]].Request_Closing_In_Hr
s3.head()


s4 = sample_data[sample_data['Complaint Type'] == top5_complaints_type_names[3]].Request_Closing_In_Hr
s4.head()


s5 = sample_data[sample_data['Complaint Type'] == top5_complaints_type_names[4]].Request_Closing_In_Hr
s5.head()

print(s1.isnull().sum())
print(s2.isnull().sum())
print(s3.isnull().sum())
print(s4.isnull().sum())
print(s5.isnull().sum())

top5_location = df_nyc['City'].value_counts()[:5]
print(top5_location)


top5_location_names = top5_location.index
print(top5_location_names)


sample_data_location_c_type = df_nyc.loc[(df_nyc['Complaint Type'].isin(top5_complaints_type_names)) & (df_nyc['City'].isin(top5_location_names)), ['Complaint Type', 'City']]
print(sample_data_location_c_type.head())


print(pd.crosstab(sample_data_location_c_type['Complaint Type'], sample_data_location_c_type['City'], margins=True))


ch2, p_value, df, exp_frq = stats.chi2_contingency(pd.crosstab(sample_data_location_c_type['Complaint Type'], sample_data_location_c_type['City']))

print(ch2)
print(p_value)

#We can see pvalue is less than 0.05 so we reject null hypothesis means complain type and location is not independent.
