import tensorflow as tf


dataTestDir = "./DATA/Testing"
dataTrainDir = "./DATA/Training"
imgSize = (512, 512)
batchSize = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataTrainDir,
    seed = 1000,
    shuffle = True,
    image_size = imgSize,
    batch_size = batchSize
)
