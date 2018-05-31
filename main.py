import os
import numpy as np
from keras.optimizers import Adam, RMSprop
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
import gc
from database.readdb import get_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


text_length = 40
batch_size = 5

X_data, Y_data, words, word2idfunc = get_data(poetry_file=os.path.join(os.getcwd(), "database", "json"),
                                              batch_size=batch_size, poet_index=2)
vocab_size = len(words) + 1

# input_shape = (text_length, vocab_size)

main_input_g = Input(shape=(text_length, vocab_size))
# embedding_layer = Embedding(vocab_size, 300, input_length=text_length)
embedding_layer = TimeDistributed(Dense(300))
x = embedding_layer(main_input_g)
x = Bidirectional(LSTM(300, return_sequences=True))(x)
main_output_g = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
generator_model = Model(inputs=[main_input_g], outputs=[main_output_g])
generator_model.summary()

main_input_d = Input(shape=(text_length,vocab_size))
aux_input_d = Input(shape=(text_length,vocab_size))

filled_in = embedding_layer(main_input_d)
context = embedding_layer(aux_input_d)
combined = Concatenate(axis=-1)([filled_in, context])
x = Bidirectional(LSTM(300, return_sequences=True))(combined)
main_output_d = TimeDistributed(Dense(1, activation='sigmoid'))(x) #or sigmoid
discriminator_model = Model(inputs = [main_input_d, aux_input_d], outputs=[main_output_d])
discriminator_model.summary()

#AM = Sequential()
optimizer = RMSprop(lr=0.0001, decay=3e-8)
AM_input = Input(shape=(text_length,vocab_size))
generator_output = generator_model(AM_input)
discriminator_output = discriminator_model([generator_output, AM_input])
AM = Model(inputs=[AM_input], outputs=discriminator_output)
#AM.add(discriminator_model)
AM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
AM.summary()


def train(train_steps=2000, batch_size=10, save_interval=50):

    for i in range(train_steps):
        if i%2 == 0:
            gc.collect()
        for j in range(10):
            indices_fake = np.random.randint(0, X_data.shape[0], size=batch_size)
            fake_unfilled = X_data[indices_fake,:]
            #noise = np.random.normal(0, 1, size=[batch_size, 100])
            fake_filled = generator_model.predict(fake_unfilled)

            indices_real = np.random.randint(0, X_data.shape[0], size=batch_size)
            real_unfilled = X_data[indices_real, :]
            real_filled = Y_data[indices_real, :]

            x_unfilled = np.concatenate((real_unfilled, fake_unfilled))
            x_filled = np.concatenate((real_filled, fake_filled))
            y = np.ones([2*batch_size, text_length, 1])
            y[batch_size:, :, :] = 0
            d_loss = discriminator_model.train_on_batch([x_filled, x_unfilled], y)
            if d_loss[0] < 0.71:
                break
        #above here is training the discriminator (DM)
        #under here is training the generator (AM)
        for j in range(10):
            y = np.ones([batch_size, text_length, 1])
            #noise = np.random.normal(0, 1, size=[batch_size, 100])
            indices_fake = np.random.randint(0, X_data.shape[0], size=batch_size)
            fake_unfilled = X_data[indices_fake, :]
            a_loss = AM.train_on_batch(fake_unfilled, y)
            if a_loss[0] < 1.2:
                break
        log_mesg = "*%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)

        if save_interval>0:
            if (i+1)%save_interval == 0:
                AM.save("AM_save.hdf5")
                discriminator_model.save("DM_save.hdf5")


if __name__=="__main__":
    optimizer = RMSprop(lr=0.0002, decay=6e-8)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train(train_steps=10000)