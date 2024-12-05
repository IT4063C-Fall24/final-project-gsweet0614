#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# The problem I am attempting to address is what is causing social media related anxiety. 
# That problem is important because mental health of especially younger people seems
# to be more and more of an issue over time.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# Is age a factor with social media usage?
# Is amount of time spent on social media mental health related?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# In people within x age gap, with x amount of time spent on social media, social media is causing a increase/decrease/no change in mental health

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[ ]:


# Start your code here

import pandas as pd

df1 = pd.read_csv('social-media.csv')

print(df1)

df2 = pd.read_csv('Time-Wasters on Social Media.csv')

print(df2)

#Seaborn or MatPlotLib didn't want to seem to play nice with Excel, so I converted the Excel file to a CSV.

df3 = pd.read_excel('Social Meida Dataset.xlsx')

df3.to_csv ('Social Media Excel Dataset', index = None, header=True)

df3_csv = pd.read_csv('Social Media Excel Dataset.csv')


# In[6]:


#EDA

#Checking for shape, data types, missing values, summary statistics, and number of duplicates.

dfs = [df1, df2, df3]
df_names = ['df1', 'df2', 'df3']

for i, df in enumerate(dfs):
    print(f"EDA for {df_names[i]}\n")

    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values[missing_values > 0], "\n")

    print("Shape of the DataFrame:", df.shape)
    print("Data Types:\n", df.dtypes, "\n")

    print("Summary Statistics:\n", df.describe(), "\n")

    duplicates = df.duplicated().sum()
    print(f"Number of Duplicates: {duplicates}\n")


# In[ ]:


#DF1 Analysis

# As we see social media rise, we not only see the average usage duration rise, but also the average age become higher over time, this is due to social media becoming such
# a staple in society and becoming so regular with mobile devices and interaction so it is more accessible to the older population, although the median age is only 33, there
# is a max age of 60 within the dataset, although this number could be false as the minimum age for people on social media is not 18, while social media companies have a regulation
# that make it so you have to be 18+, often times this rule isn't followed, so the mean could be significantly lower.

#-----

#DF2 Analysis

#In thsi dataset, we see a similar average age, I do find it interesting that no one who took part in this study had an income over $100,000, this could also be correlated to the
# slightly above average productivity loss being the average, does a large amount of social media time correlate to not having a higher income? Through doing this data analysis,
# I feel like you could pull the curtain back on social media usage and human welfare in general even further than what I do here.

#-----

#DF3 Analysis

#We see the same income numbers in this dataset as we did in the last, this dataset has another interesting stat though as I outline in a graph further down in the notebook,
# we see in this dataset a "Purchase Decision" and a "Amount Spent" metric, which is interesting because not only is social media causing income issues as we may have discovered
# in the previous dataset, it is also causing people to spend more money.


# In[ ]:


#Cleaning Data

#Dropping Duplicates
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)
df3_csv.drop_duplicates(inplace=True)

#Dropping Empty Values
df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3_csv.dropna(inplace=True)

#None of the datasets seem to have too many outliers or missing data types. I assume these were cleaned before they were posted to Kaggle.

#I also did some manual cleaning within Excel itself and CSV editing, there were a few column misspellings I fixed.


# In[4]:


#Using MatPlotLib

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2,figsize=(20,20))

#Visualization 1: This visualization is showing the relation between age and usage duration, as you can see in the chart, the younger people are, the higher usage duration they have.
axs[0,0].scatter(df1['Age'], df1['UsageDuraiton'], color='red')
axs[0,0].set_title('Age related to Usage Duration')
axs[0,0].set_xlabel('Age')
axs[0,0].set_ylabel('UsageDuration')

#Visualization 2: This visualization is showing the relation between number of videos watched and productivity loss, while this chart looks messy, we find that most people, no matter how many videos they have watched, believe that the videos they are watching are affecting their productivity between 5 and 6 on a scale of 1-10.
axs[0,1].scatter(df2['Number of Videos Watched'], df2['ProductivityLoss'], color='blue')
axs[0,1].set_title('Number of Videos Watched related to Productivity Loss')
axs[0,1].set_xlabel('NumberofVideosWatched')
axs[0,1].set_ylabel('ProductivityLoss')

plt.show()


# In[5]:


# Using Seaborn

import seaborn as sns
import matplotlib.pyplot as plt

#Visualization 3: This visualization shows us how many people per age are using social media.

plt.figure(figsize=(10, 6))
sns.countplot(data=df1, x='Age', palette='tab10')

plt.title('Amount of People per age using Social Media')
plt.xlabel('Age')
plt.ylabel('Number of People')

plt.show()


# In[6]:


# Using Seaborn

import seaborn as sns
import matplotlib.pyplot as plt

# Visualization 4: This one I actually think is the most interesting, according to the dataset description, the column (Purchase_Decision) is 
# whether the consumer's purchase decisions are influenced by social media (Yes/No). I think this is interesting because so much of what humans attach their
# mental health too is money, items, possessions, etc. and social media influencing how we handle one of the top sources of happiness or anxiety for most people is very telling.
# Especially considering according to this bar chart, just how many people are affected by social media.

plt.figure(figsize=(10, 6))
sns.countplot(data=df3_csv, x='Purchase_Decision', palette='tab10')

plt.title('Purchase Decisions Affected by Social Media')
plt.xlabel('Yes/No')
plt.ylabel('Number of Instances')

plt.show()


# In[8]:


import pandas.plotting as pdplot

pdplot.scatter_matrix(df1, figsize=(10, 10), diagonal='hist', alpha=0.8)


# In[14]:


#Preparing dataset for machine learning

import sklearn.model_selection as sklearn

X = df1.drop('Country', axis=1)
y = df1['Age']

#Splitting the dataset into a train and test set

random_train_set, random_test_set = sklearn.train_test_split(X, test_size=0.2, random_state=42)

print({random_train_set.shape})
print({random_test_set.shape})


# In[ ]:


#Using modules from sklearn to create a num pipeline for further analysis

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


# In[ ]:


#Linear Regression model from df1

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

df1_pred = LinearRegression.predict(df1)

mse = mean_squared_error(df1, df1_pred)

rmse = np.sqrt(mse)

print({rmse})


# In[ ]:


# Through all of the EDA and charts, I feel like I actually came to a different conclusiont than the question I had originally posed, I wanted to know about how mental health
#correlates to social media usage, but I feel like I actually answered a different problem, being how money correlates to social media, which when you think about it, is alot
# of what current society calls mental health, generally people with higher social media usage have lower levels of productivity (higher levels of productivity loss) and
# therefore, do not have higher paying jobs, which we have found in studies previously, that while the famous line is money can't buy happiness, in a lot of cases, it can buy
# things that lead to happiness.


# In[ ]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

