Housing Dataset EDA Project

Objective: Perform Exploratory Data Analysis (EDA) on the Housing dataset to understand the data, relationships, and insights for better decision-making.

Project Steps:

1. Introduction:

The dataset housing.csv contains information about various housing attributes including geographical data, household characteristics, and median house values. This EDA aims to explore patterns, distributions, correlations, and trends within the dataset.

2. Project Plan:

Step 1: Data Loading
 Load the housing dataset and inspect the first few rows to understand its structure.

Step 2: Data Exploration
 Perform basic exploratory data analysis including understanding data types, checking for missing values, and descriptive statistics.

Step 3: Data Visualization
 Visualize distributions, relationships, and correlations using charts to gain insights into various attributes.

3. Python Code for Project:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

 

# Step 1: Load the dataset

df = pd.read_csv("housing.csv")

 

# Step 2: Exploring Data

print(df.info())  # Basic information

print(df.describe())  # Summary statistics

print(df.isnull().sum())  # Check missing values

 

# Step 3: Data Visualization

 

# 1. Distribution of Median House Value

plt.figure(figsize=(8, 6))

sns.histplot(df['median_house_value'], bins=30, color='skyblue', kde=True)

plt.title("Distribution of Median House Value")

plt.xlabel("Median House Value")

plt.ylabel("Frequency")

plt.show()

 

# 2. Correlation Heatmap

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

 

# 3. Relationship between Median Income and Median House Value

plt.figure(figsize=(8, 6))

sns.scatterplot(x='median_income', y='median_house_value', data=df, color='purple')

plt.title("Relationship between Median Income and Median House Value")

plt.xlabel("Median Income")

plt.ylabel("Median House Value")

plt.show()

 

# 4. Distribution of Households

plt.figure(figsize=(8, 6))

sns.histplot(df['households'], bins=30, color='green', kde=True)

plt.title("Distribution of Households")

plt.xlabel("Number of Households")

plt.ylabel("Frequency")

plt.show()

 

# 5. Ocean Proximity Distribution

plt.figure(figsize=(8, 6))

sns.countplot(x='ocean_proximity', data=df, palette='cool')

plt.title("Distribution of Ocean Proximity")

plt.xlabel("Ocean Proximity")

plt.ylabel("Count")

plt.show()

Key Insights:

Distribution of Median House Value:
 The histogram helps visualize the distribution of house values and detect skewness or peaks.

Correlation Heatmap:
 A heatmap shows relationships between numerical features like median_income, total_rooms, households, and median_house_value.

Income vs House Value:
 The scatterplot displays how median income influences house values.

Household Distribution:
 A histogram for households provides insight into the typical size of households in the dataset.

Ocean Proximity:
 A bar chart visualizes how the proximity to the ocean influences housing count.

4. Conclusion:

This EDA provides a foundational understanding of the housing dataset. Further insights can be drawn by applying more advanced statistical methods or machine learning models for predictions and analysis.
