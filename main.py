# In[1]:
# #!/usr/bin/env python3
"""
   ##################  setps for bildenig  ########################
            1- import lib (done)
            2- load data (done)
            3- process data (done)
            4- model architecture (done)
            5- compile model (done)
            6- fit model (done)
            7- evaluate model || test data (done)
            8- predict values || woke done (done)
"""
from numpy.core.fromnumeric import shape
import pandas as pd
import  numpy as np
from tensorflow.keras import layers,Sequential
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.utils import layer_utils
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:
#load dataset
dataset = pd.read_csv('diabetes.csv')
print(dataset.head)
dataset.describe()
#o dataset file : csv comma-separated values
# In[3]:
# process data

X = dataset.copy() 
X = X.drop('Outcome',axis=1)  # x = Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age 
Y = dataset['Outcome'].copy() # y = Outcom

# In[4]:
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Input_shape = [X.shape[1]]

# Input_shape2 = X.values[0]
# print(f"input :  {np.array(Input_shape2)}")    

# In[5]:

data = dataset.copy()
# Visualize each column in a boxplot -> detection of outliers
plt.figure(figsize=(15, 10))
sns.boxplot(x="variable", y="value", data=pd.melt(data))
sns.stripplot(x="variable", y="value", data=pd.melt(data), color="orange", jitter=0.2, size=2.5)
plt.title("Outliers", loc="left")
plt.show()

# In[6]:
# number of rows and Columns in this dataset
# dataset.shape
# getting the statistical measures of the data
dataset.describe()

dataset['Outcome'].value_counts()
dataset.groupby('Outcome').mean()

#  In[7]:
# print(f" x shape : {X.shape},  x_train.shape : {x_train.shape}, x_test.shape : {x_test.shape}")


plt.figure(figsize = (20, 15))

plt.subplot(2,4,1)
plt.title('Pregnancies')
sns.distplot(data['Pregnancies'], kde = True, color = 'lime' )

plt.subplot(2,4,2)
plt.title('Glucose')
sns.distplot(data['Glucose'], kde = True, color = 'dodgerblue' )

plt.subplot(2,4,3)
plt.title('Blood Pressure ')
sns.distplot(data['BloodPressure'], kde = True, color = 'r' )

plt.subplot(2,4,4)
plt.title('BMI')
sns.distplot(data['BMI'], kde = True, color = 'y' )

plt.subplot(2,4,5)
plt.title('SkinThickness')
sns.distplot(data['SkinThickness'], kde = True, color = 'pink' )

plt.subplot(2,4,6)
plt.title('Insulin')
sns.distplot(data['Insulin'], kde = True, color = 'orange' )

plt.subplot(2,4,7)
plt.title('DiabetesPedigreeFunction')
sns.distplot(data['DiabetesPedigreeFunction'], kde = True, color = 'purple' )

plt.subplot(2,4,8)
plt.title('Age')
sns.distplot(data['Age'], kde = True, color = 'c' )

plt.show()

# In[8]:
model = Sequential()

model.add(layers.Dense(800 , input_dim=8 ,kernel_initializer='random_uniform', input_shape=Input_shape,activation='relu')) # input leyers 
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(250, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) # accuracy : صحه البيانات
# plt.plot(model , to_file="model_plot.png" , show_shapes=True ,show_layer_names=True)
model.fit(x_train, y_train, epochs = 20, batch_size=100, validation_data=(x_test, y_test))

output = model.evaluate(x_train,y_train)


# print(f"Xtrin :\n {x_train}     \n \n \n    y_train: \n {y_train}")
# In[9]:
# chack result
#
# if round(output[1]) == 1:
#     print("vist doctor")
#     print(output[1])
# else:  
#     print("you are good")
#     print(output[1])    
# %%
