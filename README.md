# House Prices Prediction Using TensorFlow
#### Train a Random Forest model using TensorFlow Decision Forests on the House Prices dataset

# Introduction
Predicting house prices is a crucial task as it can offer insights into the economy, inflation, consumption, and demand and supply. With the significant role played by the housing market in the business cycle, studying housing sales and prices is essential. Machine learning has emerged as a popular approach to forecast house prices based on their attributes, enabling policymakers and economists to design better policies and estimate prepayments, housing mortgage, and affordability. Generally, predicting house prices involves a regression problem that machine learning models can address effectively.

# Method
Using the code skeleton provided by [TensorFlow.org](https://www.tensorflow.org/decision_forests), the Random Forest model was built and evaluated.
```
# Install TF-DF
!pip install tensorflow tensorflow_decision_forests

# Load TF-DF
import tensorflow_decision_forests as tfdf
import pandas as pd

# Load a dataset in a Pandas dataframe.
train_df = pd.read_csv("project/train.csv")
test_df = pd.read_csv("project/test.csv")

# Convert the dataset into a TensorFlow dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="my_label")

# Train a Random Forest model.
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Summary of the model structure.
model.summary()

# Evaluate the model.
model.evaluate(test_ds)

# Export the model to a SavedModel.
model.save("project/model")
```
# Summary
