#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# We want to use dataframes once again to store and manipulate the data.

# In[2]:


pip install pandas


# In[3]:


import pandas as pd


# ---
# 
# ## Section 2 - Data loading
# 
# Similar to before, let's load our data from Google Drive for the 3 datasets provided. Be sure to upload the datasets into Google Drive, so that you can access them here.

# In[4]:


path = "/content/drive/MyDrive/Forage - Cognizant AI Program/Task 3/Resources/"

sales_df = pd.read_csv(f"{path}sales.csv")
sales_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
sales_df.head()


# In[5]:


stock_df = pd.read_csv(f"{path}sensor_stock_levels.csv")
stock_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
stock_df.head()


# In[6]:


temp_df = pd.read_csv(f"{path}sensor_storage_temperature.csv")
temp_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
temp_df.head()


# We can use the `.info()` method to look at data types.

# In[7]:


sales_df.info()


# In[8]:


stock_df.info()


# In[9]:


temp_df.info()


# Everything looks fine for the 3 datasets apart from the `timestamp` column in each dataset. Using the same helper function as before, let's convert this to the correct type for each dataset.

# In[10]:


def convert_to_datetime(data: pd.DataFrame = None, column: str = None):

  dummy = data.copy()
  dummy[column] = pd.to_datetime(dummy[column], format='%Y-%m-%d %H:%M:%S')
  return dummy


# In[11]:


sales_df = convert_to_datetime(sales_df, 'timestamp')
sales_df.info()


# In[22]:


stock_df = convert_to_datetime(stock_df, 'timestamp')
stock_df.info()


# In[13]:


temp_df = convert_to_datetime(temp_df, 'timestamp')
temp_df.info()


# This looks much better!
# 
# ---
# 
# ## Section 5 - Merge data
# 
# Currently we have 3 datasets. In order to include all of this data within a predictive model, we need to merge them together into 1 dataframe. 
# 
# If we revisit the problem statement: 
# 
# ```
# “Can we accurately predict the stock levels of products, based on sales data and sensor data, 
# on an hourly basis in order to more intelligently procure products from our suppliers.”
# ```
# 
# The client indicates that they want the model to predict on an hourly basis. Looking at the data model, we can see that only column that we can use to merge the 3 datasets together is `timestamp`.
# 
# So, we must first transform the `timestamp` column in all 3 datasets to be based on the hour of the day, then we can merge the datasets together.

# In[14]:


sales_df.head()


# In[25]:


from datetime import datetime

def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
  dummy = data.copy()
  new_ts = dummy[column].tolist()
  new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
  new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
  dummy[column] = new_ts
  return dummy


# In[26]:


sales_df = convert_timestamp_to_hourly(sales_df, 'timestamp')
sales_df.head()


# In[27]:


stock_df = convert_timestamp_to_hourly(stock_df, 'timestamp')
stock_df.head()


# In[28]:


temp_df = convert_timestamp_to_hourly(temp_df, 'timestamp')
temp_df.head()


# Now you can see all of the `timestamp` columns have had the minutes and seconds reduced to `00`. The next thing to do, is to aggregate the datasets in order to combine rows which have the same value for `timestamp`.
# 
# For the `sales` data, we want to group the data by `timestamp` but also by `product_id`. When we aggregate, we must choose which columns to aggregate by the grouping. For now, let's aggregate quantity.

# In[45]:


sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
sales_agg.head()


# We now have an aggregated sales data where each row represents a unique combination of hour during which the sales took place from that weeks worth of data and the product_id. We summed the quantity and we took the mean average of the unit_price.
# 
# For the stock data, we want to group it in the same way and aggregate the `estimated_stock_pct`.

# In[46]:


stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
stock_agg.head()


# This shows us the average stock percentage of each product at unique hours within the week of sample data.
# 
# Finally, for the temperature data, product_id does not exist in this table, so we simply need to group by timestamp and aggregate the `temperature`.

# In[47]:


temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()
temp_agg.head()


# This gives us the average temperature of the storage facility where the produce is stored in the warehouse by unique hours during the week. Now, we are ready to merge our data. We will use the `stock_agg` table as our base table, and we will merge our other 2 tables onto this.

# In[48]:


merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
merged_df.head()


# In[50]:


merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
merged_df.head()


# In[51]:


merged_df.info()


# We can see from the `.info()` method that we have some null values. These need to be treated before we can build a predictive model. The column that features some null values is `quantity`. We can assume that if there is a null value for this column, it represents that there were 0 sales of this product within this hour. So, lets fill this columns null values with 0, however, we should verify this with the client, in order to make sure we're not making any assumptions by filling these null values with 0.

# In[52]:


merged_df['quantity'] = merged_df['quantity'].fillna(0)
merged_df.info()


# We can combine some more features onto this table too, including `category` and `unit_price`.

# In[58]:


product_categories = sales_df[['product_id', 'category']]
product_categories = product_categories.drop_duplicates()

product_price = sales_df[['product_id', 'unit_price']]
product_price = product_price.drop_duplicates()


# In[59]:


merged_df = merged_df.merge(product_categories, on="product_id", how="left")
merged_df.head()


# In[60]:


merged_df = merged_df.merge(product_price, on="product_id", how="left")
merged_df.head()


# In[61]:


merged_df.info()


# Now we have our table with 2 extra features!
# 
# ---
# 
# ## Section 6 - Feature engineering
# 
# We have our cleaned and merged data. Now we must transform this data so that the columns are in a suitable format for a machine learning model. In other terms, every column must be numeric. There are some models that will accept categorical features, but for this exercise we will use a model that requires numeric features.
# 
# Let's first engineer the `timestamp` column. In it's current form, it is not very useful for a machine learning model. Since it's a datetime datatype, we can explode this column into day of week, day of month and hour to name a few.

# In[70]:


merged_df['timestamp_day_of_month'] = merged_df['timestamp'].dt.day
merged_df['timestamp_day_of_week'] = merged_df['timestamp'].dt.dayofweek
merged_df['timestamp_hour'] = merged_df['timestamp'].dt.hour
merged_df.drop(columns=['timestamp'], inplace=True)
merged_df.head()


# The next column that we can engineer is the `category` column. In its current form it is categorical. We can convert it into numeric by creating dummy variables from this categorical column.
# 
# A dummy variable is a binary flag column (1's and 0's) that indicates whether a row fits a particular value of that column. For example, we can create a dummy column called category_pets, which will contain a 1 if that row indicates a product which was included within this category and a 0 if not.

# In[71]:


merged_df = pd.get_dummies(merged_df, columns=['category'])
merged_df.head()


# In[72]:


merged_df.info()


# Looking at the latest table, we only have 1 remaining column which is not numeric. This is the `product_id`.
# 
# Since each row represents a unique combination of product_id and timestamp by hour, and the product_id is simply an ID column, it will add no value by including it in the predictive model. Hence, we shall remove it from the modeling process.

# In[73]:


merged_df.drop(columns=['product_id'], inplace=True)
merged_df.head()


# This feature engineering was by no means exhaustive, but was enough to give you an example of the process followed when engineering the features of a dataset. In reality, this is an iterative task. Once you've built a model, you may have to revist feature engineering in order to create new features to boost the predictive power of a machine learning model.
# 
# ---
# 
# ## Section 7 - Modelling
# 
# Now it is time to train a machine learning model. We will use a supervised machine learning model, and we will use `estimated_stock_pct` as the target variable, since the problem statement was focused on being able to predict the stock levels of products on an hourly basis.
# 
# Whilst training the machine learning model, we will use cross-validation, which is a technique where we hold back a portion of the dataset for testing in order to compute how well the trained machine learning model is able to predict the target variable.
# 
# Finally, to ensure that the trained machine learning model is able to perform robustly, we will want to test it several times on random samples of data, not just once. Hence, we will use a `K-fold` strategy to train the machine learning model on `K` (K is an integer to be decided) random samples of the data.
# 
# First, let's create our target variable `y` and independent variables `X`

# In[75]:


X = merged_df.drop(columns=['estimated_stock_pct'])
y = merged_df['estimated_stock_pct']
print(X.shape)
print(y.shape)


# This shows that we have 29 predictor variables that we will train our machine learning model on and 10845 rows of data.
# 
# Now let's define how many folds we want to complete during training, and how much of the dataset to assign to training, leaving the rest for test.
# 
# Typically, we should leave at least 20-30% of the data for testing.

# In[80]:


K = 10
split = 0.75


# For this exercise, we are going to use a `RandomForestRegressor` model, which is an instance of a Random Forest. These are powerful tree based ensemble algorithms and are particularly good because their results are very interpretable.
# 
# We are using a `regression` algorithm here because we are predicting a continuous numeric variable, that is, `estimated_stock_pct`. A `classification` algorithm would be suitable for scenarios where you're predicted a binary outcome, e.g. True/False.
# 
# We are going to use a package called `scikit-learn` for the machine learning algorithm, so first we must install and import this, along with some other functions and classes that can help with the evaluation of the model.

# In[77]:


get_ipython().system('pip install scikit-learn')


# In[89]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# And now let's create a loop to train `K` models with a 75/25% random split of the data each time between training and test samples

# In[95]:


accuracy = []

for fold in range(0, K):

  # Instantiate algorithm
  model = RandomForestRegressor()
  scaler = StandardScaler()

  # Create training and test samples
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

  # Scale X data, we scale the data because it helps the algorithm to converge
  # and helps the algorithm to not be greedy with large values
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  # Train model
  trained_model = model.fit(X_train, y_train)

  # Generate predictions on test sample
  y_pred = trained_model.predict(X_test)

  # Compute accuracy, using mean absolute error
  mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
  accuracy.append(mae)
  print(f"Fold {fold + 1}: MAE = {mae:.3f}")

print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")


# Note, the output of this training loop may be slightly different for you if you have prepared the data differently or used different parameters!
# 
# This is very interesting though. We can see that the `mean absolute error` (MAE) is almost exactly the same each time. This is a good sign, it shows that the performance of the model is consistent across different random samples of the data, which is what we want. In other words, it shows a robust nature.
# 
# The `MAE` was chosen as a performance metric because it describes how closely the machine learning model was able to predict the exact value of `estimated_stock_pct`.
# 
# Even though the model is predicting robustly, this value for MAE is not so good, since the average value of the target variable is around 0.51, meaning that the accuracy as a percentage was around 50%. In an ideal world, we would want the MAE to be as low as possible. This is where the iterative process of machine learning comes in. At this stage, since we only have small samples of the data, we can report back to the business with these findings and recommend that the the dataset needs to be further engineered, or more datasets need to be added.
# 
# As a final note, we can use the trained model to intepret which features were signficant when the model was predicting the target variable. We will use `matplotlib` and `numpy` to visualuse the results, so we should install and import this package.

# In[103]:


pip install matplotlib
pip install numpy


# In[104]:


import matplotlib.pyplot as plt
import numpy as np


# In[105]:


features = [i.split("__")[0] for i in X.columns]
importances = model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# This feature importance visualisation tells us:
# 
# - The product categories were not that important
# - The unit price and temperature were important in predicting stock
# - The hour of day was also important for predicting stock
# 
# With these insights, we can now report this back to the business
