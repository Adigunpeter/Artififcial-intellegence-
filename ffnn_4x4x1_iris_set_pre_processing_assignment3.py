# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('D_iris2_train0.csv', delimiter=',')
print (dataset)
# split into input (X) and output (y) variables
for i in range(10):
  X = dataset[113*i:113*(i+1),0:4]
  y = dataset[113*i:113*(i+1),4:7]
  print(X)
  print()

  print(y)
"""
# define the keras model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=10)
# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
#accuracy = model.evaluate(X, y)
model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))
"""