import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image, ImageEnhance

from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.metrics import *
from keras.optimizers import *
from keras.applications import *
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random

from flask import Flask, render_template, request
import os



trainDataset = ['deployment/DATA/Training/']
testDataset = ['deployment/DATA/Testing/']

trainPaths = []
trainLables = []

for trainDir in trainDataset:
    for label in os.listdir(trainDir):
        for image in os.listdir(trainDir+label):
            trainPaths.append(trainDir+label+'/'+image)
            trainLables.append(label)

trainPaths, trainLables = shuffle(trainPaths, trainLables)

testPaths = []
testLables = []

for testDir in testDataset:
    for label in os.listdir(testDir):
        for image in os.listdir(testDir+label):
            testPaths.append(testDir+label+'/'+image)
            testLables.append(label)

testPaths, testLables = shuffle(testPaths, testLables)
uniqueLables = os.listdir(trainDir)


agu = 0.5
def augmentImages(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(1-agu,1+agu))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(1-agu,1+agu))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(1-agu,1+agu))
    
    image = np.array(image)/255.0
    return image
    
imageSize = 128


        # FOR VISULISIN THE SET
def openImages(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(imageSize,imageSize))
        image = augmentImages(image)
        images.append(image)
    return np.array(images)

def encodeLables(labels):
    encoded = []
    for x in labels:
        encoded.append(uniqueLables.index(x))
    return np.array(encoded)

def decodeLables(labels):
    decoded = []
    for x in labels:
        decoded.append(uniqueLables[x])
    return np.array(decoded)



try:
    model = tf.keras.models.load_model('model.keras')
    print("LOADED MODEL")
except Exception as e:
    print(e)

    # images = openImages(trainPaths[40:49])
    # lbl = trainLables[40:49]
    # fig = plt.figure(figsize=(12, 6))
    # for x in range(1, 9):
    #     fig.add_subplot(2, 4, x)
    #     plt.axis('off')
    #     plt.title(lbl[x])
    #     plt.imshow(images[x])
    # plt.rcParams.update({'font.size': 12})
    # plt.show()


    def datagen(paths, labels, batchSize=12, epochs=1):
        for _ in range(epochs):
            for x in range(0, len(paths), batchSize):
                batchPaths = paths[x:x+batchSize]
                batchImages = openImages(batchPaths)
                batchLabels = labels[x:x+batchSize]
                batchLabels = encodeLables(batchLabels)
                yield batchImages, batchLabels

    baseModel = VGG16(input_shape=(imageSize,imageSize,3), include_top=False, weights='imagenet')
    for layer in baseModel.layers:
        layer.trainable = False
    # Set the last vgg block to trainable
    baseModel.layers[-2].trainable = True
    baseModel.layers[-3].trainable = True
    baseModel.layers[-4].trainable = True

    model = Sequential()
    model.add(Input(shape=(imageSize,imageSize,3)))
    model.add(baseModel)
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(uniqueLables), activation='softmax'))


    model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])
    
    batchSize = 25
    steps = int(len(trainPaths)/batchSize)
    epochs = 1
    history = model.fit(datagen(trainPaths, trainLables, batchSize=batchSize, epochs=epochs),
                        epochs=epochs, steps_per_epoch=steps)
    model.save("model.keras")
    print("YOU NEED TO RENAME MODEL2.KERAS TO MODEL.KERAS")

    # steps = int(len(testPaths)/batchSize)
    # yPred = []
    # yTrue = []
    # for x,y in tqdm(datagen(testPaths, testLables, batchSize=batchSize, epochs=1), total=steps):
    #     pred = model.predict(x)
    #     pred = np.argmax(pred, axis=-1)
    #     for i in decodeLables(pred):
    #         yPred.append(i)
    #     for i in decodeLables(y):
    #         yTrue.append(i)


###################################################
            # AFTER ALL THAT #
###################################################





# to see the confusion matrix run the code below
            
# minSamplesPerClass = min([testLables.count(label) for label in uniqueLables])
# balancedTestPaths = []
# balancedTestLables = []

# for label in uniqueLables:
#     classPaths = [path for path, lbl in zip(testPaths, testLables) if lbl == label]
#     balancedTestPaths.extend(classPaths[:minSamplesPerClass])
#     balancedTestLables.extend([label] * minSamplesPerClass)


# testImagesBalanced = openImages(balancedTestPaths)
# testLables_encoded_balanced = encodeLables(balancedTestLables)

# predictions_balanced = model.predict(testImagesBalanced)
# predicted_labels_balanced = np.argmax(predictions_balanced, axis=1)

# confusionMatrixBalanced = tf.math.confusion_matrix(testLables_encoded_balanced, predicted_labels_balanced, num_classes=len(uniqueLables))

# plt.figure(figsize=(8, 6))
# sns.heatmap(confusionMatrixBalanced, annot=True, fmt='d', cmap='Blues', xticklabels=uniqueLables, yticklabels=uniqueLables)
# plt.title('Confusion Matrix (Balanced Testing)')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()


###################################
            #prediction
###################################
def DECLB(label): # decode lables
    return uniqueLables[label]

def prd(path):
    image = load_img(path, target_size=(imageSize,imageSize))
    image = augmentImages(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predictedLabel = DECLB(np.argmax(prediction))
    certainty = round(np.max(prediction) * 100)
    return predictedLabel, certainty


###################################
        # web server #
###################################


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
    if request.method == 'POST':
        uploadedFile = request.files['file']
        if uploadedFile:
            tempDir = os.path.join(os.path.dirname(__file__), 'temp')
            if not os.path.exists(tempDir):
                os.makedirs(tempDir)
            loco = os.path.join(tempDir, uploadedFile.filename)
            uploadedFile.save(loco)
            prdLabel, Cert = prd(loco)
            return f'this is a {prdLabel}, with {Cert}% chance'
        else:
            return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)
