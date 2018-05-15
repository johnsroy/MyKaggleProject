# Identify credit card fraudulent transactions

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 15, y = 15, input_len = 30, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) 
colorbar()
markers = ['o', 's']
colors = ['r', 'g'] 


for i, x in enumerate(X):
    w = som.winner(x) # we are getting the winning node of each customer
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
#The following coordinates must be manually updated each time based on SOM coordinates since the random state parameter is not defined
frauds = np.concatenate((mappings[(3,11)],mappings[(8,1)], mappings[(10,3)], mappings[(13,1)]), axis = 0)
frauds = sc.inverse_transform(frauds) #7404 frauds

### Creating the hybrid model ###
customers = dataset.iloc[:, 1:-1].values

# Creating the dependent variable

#is_fraud = np.zeros(len(dataset))
#for i in range(len(dataset)):
#    if dataset.iloc[i,0] in frauds:
#        is_fraud[i] = 1
is_fraud = y #Simpler

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and not adding any hidden layers
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 3)

# Predicting the probabilities of frauds and sorting them
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]



