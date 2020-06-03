import dataset
import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE = 50       #resolution to which images are resized
ALPHA = .0001        #learning rate

trainDirectory = "./training_data"
testDirectory = "./testing_data"

#define name for the model, change if # of convolutional layers is changed
modelName = "corgiClass-{}--{}".format(ALPHA, '4_conv_layers_final')

#define training dataset, 20% are for validation
dSet = dataset.read_train_sets("./data/training_data", IMG_SIZE, ['pembroke', 'cardigan'], .2)

#define test dataset, since its the test, 0 are for validation
testSet = dataset.read_train_sets("./data/training_data", IMG_SIZE, ['pembroke', 'cardigan'], 0)

# Building convolutional neural net

#input layer
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

#convolutional layers
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

#
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

#output Layer
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=ALPHA, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

#loads model if one exists
if os.path.exists('{}.meta'.format(modelName)):
    model.load(modelName)
    print("previous model loaded")

train = dSet.train
validate = dSet.valid
X = np.array(([i for i in train.images()])).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = [i for i in train.labels()]
validateX = np.array(([i for i in validate.images()])).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
validateY = [i for i in validate.labels()]

#function to train the model, uncomment if you want it to train more
#model.fit({'input': X}, {'targets': Y}, n_epoch = 20, validation_set = ({'input': validateX}, {'targets': validateY}), snapshot_step = 500, show_metric = True, run_id =modelName)


#save model
model.save(modelName)

#count for correct prediction
correctCount = 0

for i in range(testSet.train.num_examples()):
    prediction = model.predict(testSet.train.images()[i].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
    print("prediction is: ", prediction, "\tactual val is : ", testSet.train.labels()[i])
    #prediction is the largest
    if np.argmax(prediction) == np.argmax(testSet.train.labels()[i]):
        correctCount += 1
print("testing accuracy = ", correctCount/testSet.train.num_examples())
