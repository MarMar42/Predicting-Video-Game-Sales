import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

df = pd.read_csv('vgsales-12-4-2019-short.csv', usecols = ['Name','Genre','Platform','Publisher','Developer','Year','Global_Sales'])

#Drop all records with no values
df.dropna(inplace = True)
#Only select records which with global sales greater than 0
df = df[df['Global_Sales'] > 0]
#Normalize Year
df['Year'] = (df['Year']- df['Year'].min())/(df['Year'].max()-df['Year'].min())
df = pd.get_dummies(df, columns=['Genre','Platform','Publisher','Developer'],prefix= '', prefix_sep='')
#Sort columns
ls = []
for i in range(len(df.columns)):
    if i != 1:
        ls.append(df.columns[i])
ls.append(df.columns[1])
df = df[ls]

#Split data into features and target
x = df[ls[1:-1]].to_numpy()[:,np.newaxis]
y = df[ls[-1]].to_numpy().reshape(-1,1)
#Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)#Validation data of 10% (15% of 67%)
np.random.seed(42)
#Initialise Neural Network
net = Network()
n_hidden_nodes = 500 
#Input layer
net.add(FCLayer(x.shape[-1], n_hidden_nodes))
net.add(ActivationLayer(tanh, tanh_prime))
#hidden layers
net.add(FCLayer(n_hidden_nodes, n_hidden_nodes))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(n_hidden_nodes, n_hidden_nodes))
net.add(ActivationLayer(tanh, tanh_prime))
#Output layer
net.add(FCLayer(n_hidden_nodes, 1))
#Took out activation function here
net.use(mse, mse_prime)
net.fit(x_train,y_train, epochs = 100, learning_rate = 0.001)

y_pred = np.array(net.predict(x_test)).reshape(-1,1)

np.savetxt('data_test500Trial.csv', y_pred, delimiter=',')
y_pred = np.array(net.predict(x_train)).reshape(-1,1)
np.savetxt('data_train500Trial.csv', y_pred,delimiter =',')
y_pred = np.array(net.predict(x_val)).reshape(-1,1)
np.savetxt('data_val500Trial.csv', y_pred, delimiter =',')
