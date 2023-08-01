#import libraries
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data()

#padding
w = 30 #all reviews should have exactly "w" words
x_train = pad_sequences(x_train, maxlen=w)
x_test = pad_sequences(x_test, maxlen=w)

#Build the RNN model
model = Sequential()

#add the layers
#input shape = (w, 1)
model.add(SimpleRNN(32, input_shape=(w,1), return_sequences=False)) #recurrent layer with 32 neurons
model.add(Dense(1,activation='sigmoid')) #output layer with 1 neuron    

#compile the model
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training and testing
epochs = 5
model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))
    