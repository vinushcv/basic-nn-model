# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
developed by: Vinush.cv

Reg no: 212222230176

```python
import pandas as pd
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)

worksheet=gc.open('deeplearning').sheet1
data=worksheet.get_all_values()

data1=pd.DataFrame(data[1:],columns=data[0])
data1=data1.astype({'input':'float'})
data1=data1.astype({'output':'float'})
data1.head()

x=data1[['input']].values
y=data1[['output']].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

scaler.fit(x_train)

x_train1=scaler.transform(x_train)


from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

ai_brain=Sequential([Dense(7,activation="relu"),Dense(14,activation="relu"),Dense(1)])

ai_brain.compile(optimizer="rmsprop",loss="mse")


ai_brain.fit(x_train,y_train,epochs=3000)


loss=pd.DataFrame(ai_brain.history.history)

loss.plot()

x_test1=scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)

x_n1=[[11]]
x_n1_1=scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```

## Dataset Information

![image](https://github.com/vinushcv/basic-nn-model/assets/113975318/6d643ed8-da2b-448c-8b2b-3941f947a890)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/vinushcv/basic-nn-model/assets/113975318/d454b68a-5b40-4866-bb9c-e5b60db2ea7a)


### Test Data Root Mean Squared Error

![image](https://github.com/vinushcv/basic-nn-model/assets/113975318/c6d93ecb-58c0-499a-bffa-039ff9fce1dd)


### New Sample Data Prediction

  ![image](https://github.com/vinushcv/basic-nn-model/assets/113975318/0bee52f9-3bd6-4abe-857c-f0fe968d85e0)



## RESULT

Thus a neural network regression model for the given dataset is executed successfully.

