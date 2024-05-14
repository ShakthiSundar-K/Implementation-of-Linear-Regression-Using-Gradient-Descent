# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a company using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary libraries.
2.Define linear regression function.
3.Begin by reading the dataset using the read_csv function. 
4.Apply the linear_regression function to the standardized input features X1_Scaled and target variable Y1_Scaled to obtain the optimal parameters theta.
5.Prepare new data and make predictions using the trained model.
6.Print the predicted value obtained from the regression analysis.
```

## Program:
``` py
Developed by: SHAKTHI SUNDAR K
RegisterNumber: 212222040152

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term 
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta

data = pd.read_csv('/content/50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding column 
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
### DATA:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116143/ffc0b251-fd0f-4bbb-8427-3e7ceac8aa88)

![image](https://github.com/ShakthiSundar-K/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116143/a879e9b0-7753-4df0-89f9-f062ccc25120)

![image](https://github.com/ShakthiSundar-K/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116143/f93c11cd-cc0b-4698-b57b-cf7282b87134)
### PREDICTED VALUE:
![image](https://github.com/ShakthiSundar-K/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116143/5d0b22d2-5d7c-4622-8e5d-38592f2f134d)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
