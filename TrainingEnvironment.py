from keras.layers import Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt

from keras import regularizers

#
# BUILDING THE MODELS
# Size of our encoded representation
encoding_dim2 = 32 

input_img2 = Input(shape=(784,))

# Encoding the image down to a 32-dim vector
encoded2 = Dense(512, activation='relu')(input_img2)
encoded2 = Dense(256, activation='relu')(encoded2)
encoded2 = Dense(128, activation='relu')(encoded2)
encoded2 = Dense(64, activation='relu')(encoded2)
encoded2 = Dense(encoding_dim2, activation='relu')(encoded2)

# Decoding the encoded image back to a 784-dim vector
decoded2 = Dense(64, activation='relu')(encoded2)
decoded2 = Dense(128, activation='relu')(decoded2)
decoded2 = Dense(256, activation='relu')(decoded2)
decoded2 = Dense(512, activation='relu')(decoded2)
decoded2 = Dense(784, activation='sigmoid')(decoded2)

# Model 1: The entire autoencoder mapped from input to ouput
autoencoder2 = Model(input_img2, decoded2)

# Model 2: The encoder part only
encoder2 = Model(input_img2, encoded2)

# create a placeholder for an encoded (32-dimensional) input
encoded_input2 = Input(shape=(encoding_dim2,))

# retrieve the decoding layers to recursively build the decoder part
decoder_layer2_5 = autoencoder2.layers[-5]
decoder_layer2_4 = autoencoder2.layers[-4]
decoder_layer2_3 = autoencoder2.layers[-3]
decoder_layer2_2 = autoencoder2.layers[-2]
decoder_layer2_1 = autoencoder2.layers[-1]

# Model 3: The decoder part only
decoder2 = Model(encoded_input2, decoder_layer2_1(decoder_layer2_2(decoder_layer2_3(decoder_layer2_4(decoder_layer2_5(encoded_input2))))))

autoencoder2.compile(optimizer='adadelta', loss="binary_crossentropy")

from keras.datasets import mnist
import numpy as np
(x_train, train_labels), (x_test, test_labels) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Train the main model for 1000 epochs
# Takes about 50 mins in colaboratory
# We will export this model to the testing enviroment later
autoencoder2.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                # Display only for validation loss val_loss
                validation_data=(x_test, x_test))

# encode and decode some digits
encoded_imgs2 = encoder2.predict(x_test)
decoded_imgs2 = decoder2.predict(encoded_imgs2)

# Using pyplot to display n sample results
n = 10 
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#
# EXPORT THE MODELS USING JSON
#serialize model to JSON
model_json = autoencoder2.to_json()
with open("autoencoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder2.save_weights("autoencoder.h5")
print("Saved model to disk")

# serialize model to JSON
model_json = encoder2.to_json()
with open("encoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder2.save_weights("encoder.h5")
print("Saved model to disk")

# serialize model to JSON
model_json = decoder2.to_json()
with open("decoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
decoder2.save_weights("decoder.h5")
print("Saved model to disk")

# 
# DOWNLOADING THE EXPORTED MODELS 
# Save the models somewhere in Google Drive to allow to be imported to the testing environment

from google.colab import files

files.download('autoencoder.json')
files.download('encoder.json')
files.download('encoder.h5')
files.download('decoder.json')
files.download('decoder.h5')

# end of training environment
