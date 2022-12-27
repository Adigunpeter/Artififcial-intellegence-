# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# load the dataset
dataset = loadtxt('D_iris2_train1.csv', delimiter=',')
print (dataset)
# split into input (X) and output (y) variables
z = []
for i in range(10):
  X = dataset[75*i:75*(i+1),0:4]
  y = dataset[75*i:75*(i+1),4]
  #print(X)#print()#print(y)
  

  # define the keras model
  model = Sequential()
  model.add(Dense(5, input_dim=4, activation='sigmoid'))
  model.add(Dense(1, activation='sigmoid'))
  # compile the keras model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit the keras model on the dataset
  model.fit(X, y, epochs=10, batch_size=10)
  # evaluate the keras model
  _, accuracy = model.evaluate(X, y)
  accuracy = model.evaluate(X, y)
  model.evaluate(X, y)
  print('Accuracy: %.2f' % (accuracy[0]*100))
  z.append (accuracy[0])
  print (z)
plt.plot(range(10),z) 
plt.show() 