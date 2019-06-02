from keras.layers import Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt

"""
Model architecture definition

Encoder: 784->512->256->128->64->32
Decoder: 32->64->128->256->512->784

Autoencoder: Encoder->Decoder
"""
encoding_dim = 32   # array size for encoded representations

input_img = Input(shape=(784,))

# Encoding the image down to a 32-dim vector
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoding the encoded image back to a 784-dim vector
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

# Placeholder for an encoded variables
encoded_input = Input(shape=(encoding_dim,))

# retrieve the decoding layers to recursively build the decoder part
decoder_layer_5 = autoencoder.layers[-5]
decoder_layer_4 = autoencoder.layers[-4]
decoder_layer_3 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_1 = autoencoder.layers[-1]

# Model 3: The decoder part only
decoder = Model(encoded_input, decoder_layer_1(decoder_layer_2(decoder_layer_3(decoder_layer_4(decoder_layer_5(encoded_input))))))

autoencoder.compile(optimizer='adam', loss="binary_crossentropy")

from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Train the main model for 100 epochs
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                # Display only for validation loss val_loss
                validation_data=(x_test, x_test))

"""
Encoding and decoding samples
"""
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

"""
Results
"""
n = 10 
plt.figure(figsize=(10, 2))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


"""
Export models - json files
"""
from custom_functions import save_model
save_model(autoencoder, "autoencoder")
save_model(encoder, "encoder")
save_model(decoder, "decoder")