import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")
#print(passengers)

# Update sex column to numerical
passengers["Sex"] = passengers["Sex"].map({"female" : 1, "male": 0})
#print(passengers)

#Inspecting the values in the age column
print(passengers["Age"].values)

# Fill the nan values in the age column with the mean age
passengers["Age"].fillna(passengers.Age.mean(), inplace = True)
#print(passengers)

# Create a first class column
passengers["FirstClass"] = passengers["Pclass"].apply(lambda row: 1 if row == 1 else 0)
#print(passengers)

# Create a second class column
passengers["SecondClass"] = passengers["Pclass"].apply(lambda row: 1 if row == 2 else 0)
print(passengers)

# Select the desired columns to build model on
#Features
features = passengers[["Sex", "Age", "FirstClass", "SecondClass"]]
#Outcome feature
survival = passengers["Survived"]
#print(features)
#print(survival)

# Perform train, test, split on the data (splitting the data)
features_train, features_test, survival_train, survival_test = train_test_split(features, survival, test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
norm = StandardScaler()
features_train = norm.fit_transform(features_train)
features_test = norm.transform(features_test)

# Create and train the model (Logistic Regression model since the survived / not survived classification is binary)
model = LogisticRegression()
model.fit(features_train, survival_train)

# Score the model on the train data
print(model.score(features_train, survival_train))

# Score the model on the test data
print(model.score(features_test, survival_test))

# Analyze the coefficients
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Esi = np.array([1.0,24.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Esi])
#print(sample_passengers)

# Scale the sample passenger features
sample_passengers = norm.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))