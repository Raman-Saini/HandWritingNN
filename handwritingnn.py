#Import Packages
import numpy as np
import mnist #Get Data Set From 
import matplotlib.pyplot as plt #Graph
from keras.models import Sequential #ANN Architecture
from keras.layers import Dense #The Layers In The ANN
from keras.utils import to_categorical

#Load Data To Set
train_images = mnist.train_images() #Training Data - Images
train_labels = mnist.train_labels() #Training Data - Labels
test_images = mnist.test_images() #Test Data - Images
test_labels = mnist.test_labels() #Test Data - Labels

#Normalize The Images. Normalize The Pixel Values From [0, 255] 
#To [-0.5, 0.5] To Make Our Network Easier To Train
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5
#Flatten The Images - Each 28X28 Into A 784 Dimensional Vector To Pass Into NN
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
#Print The Shape
print(train_images.shape) #60,000 Rows, 784 Rows
print(test_images.shape) #10,000 Rows, 784 Rows

#Build The Model 
#With 3 Layers, 
#2 Layers With 64 Neurons And The Relu Function,
#1 Layer With 10 Neurons And Softmax Function
model = Sequential()
model.add( Dense(64, activation='relu', input_dim=784))
model.add( Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compile The Model
#The Lost Function Measures How Well The Model Did On Training 
#And Then Tries To Improve On It Using The Optimzer
model.compile(
  optimizer='adam',
    loss = 'categorical_crossentropy', #Classes That Are Greater Than 2
    metrics = ['accuracy']
)

#Train the model
model.fit(
  train_images,
    to_categorical(train_labels), # Ex. 2 it expects [0, 0 ,1,0, 0, 0,0,0,0,0]
    epochs = 5, #The number of iterations over the entire dataset to train on
    batch_size=32 #the number of samples per gradient update for training
)

#Evaluate The Model 
model.evaluate(
    test_images,
     to_categorical(test_labels)
)
#Save
#model.save_weights('Model.h5')

#Predict On The First 15 Test Images
predictions = model.predict(test_images[:15])
#Print Our Models Prediction
print(np.argmax(predictions, axis = 1))
print(test_labels[:15])

for i in range(0,15):
  first_image = test_images[i]
  first_image = np.array(first_image, dtype='float')
  pixels = first_image.reshape((28,28))
  plt.imshow(pixels)
  plt.show()
