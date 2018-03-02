# command line Script for using the prediction using trained weights
from keras.layers import RepeatVector, Dense, Activation
from keras.layers import Bidirectional, Concatenate, Dot, LSTM, Input
from keras.optimizers import Adam
from keras.models import Model

import os.path
import numpy as np
import random
from utility import *
import os


m = 1000
Tx = 30
Ty = 10 

dataset, X, Y, X_ohe, Y_ohe, human_vocab, human_char_idx, machine_vocab, machine_char_idx, machine_idx_char = create_training_data(
    m, Tx, Ty)

# shared layers
# Following layers are mainly for the neural network for finding the attention weights
concatenator = Concatenate(axis = -1)
repeator = RepeatVector(Tx)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation('softmax')
dotor = Dot(axes = 1)

# shared layers
# below layers are for the post LSTM network
n_acti_pre = 32 # hidden activation units in pre attention LSTM network
n_acti_post = 64 # hidden activation units in post attention LSTM network

# post activation LSTM cell
post_acti_LSTM = LSTM(n_acti_post, return_state = True)
output_layer = Dense(len(machine_vocab), activation = 'softmax')

# initial input states for post LSTM network
ini_acti_post = np.zeros((m, n_acti_post))
ini_mem_post = np.zeros((m, n_acti_post))

# for getting the context value using attention weights for each timestep of output
def get_context(acti_pre, acti_post_prev):
    """
    Finds the context value using attention weights for each timestep of output
    Arguments:
        acti_pre -- numpy-array(m, Tx, 2*n_acti_pre): hidden state values of  Bidirectional-LSTM network 
        acti_post_prev -- numpy array(m, n_acti_post): previous hidden state of the (post-attention) LSTM

    Returns:
        context -- vector, input of the next post-attetion LSTM cell
    """
    # repeat the previous state of the post -attention LSTM cell
    acti_post_prev = repeator(acti_post_prev)
    # concatenate the output with the activations from the different pre attetntion LSTM cells
    concat_vals = concatenator([acti_post_prev, acti_pre])
    # pass the concatenated values through a dense layer
    inter_vals = densor1(concat_vals)
    inter_vals = densor2(inter_vals)
    # pass the intermediate value through a softmax layer to get the alpha values
    alpha_vals = activator(inter_vals)
    # after getting the alpha values find the sum of weighted product of pre- activations values with their context weights(alpha values)
    context = dotor([alpha_vals, acti_pre])
    
    return context


# for creating a Keras model Instance
def create_model(human_vocab_len, machine_vocab_len, Tx, Ty, n_acti_pre, n_acti_post):
    """
    Arguments:
        human_vocab_len - length of the human_char_idx dictionary
        machine_vocab_len - length of the machine_char_idx dictionary
        Tx - length of the input sequence
        Ty - length of the output sequence
        n_acti_pre - no. of hidden state units of the Bi-LSTM
        n_acti_post - no. of hidden state units of the post-attention LSTM

    Returns:
        model -- Keras model instance
    """
    # for storing the outputs
    outputs = []
    
    # define input for the model
    X = Input(shape=(Tx, human_vocab_len))
    # for the decoder LSTM i.e post attention network
    # initial values
    ini_acti_post = Input(shape=(n_acti_post,), name='ini_acti_post')
    ini_mem_post = Input(shape=(n_acti_post,), name='ini_mem_post')
   
    
    # current timestep values
    mem_post = ini_mem_post
    acti_post = ini_acti_post
    
    # make the encoder Bidirectional LSTM network
    acti_pre = Bidirectional(LSTM(n_acti_pre, return_sequences=True))(X)
    
    # loop over each output timestep
    for timestep in range(Ty):
        # get the current context value
        context = get_context(acti_pre, acti_post)
        # feed the context value to the decoder LSTM network post attention
        acti_post, _, mem_post = post_acti_LSTM (context, initial_state=[acti_post, mem_post])
        # get the softmax output 
        target_output = output_layer(acti_post)
        
        # add the output target value
        outputs.append(target_output)
   
    # make the model and return model instance
    model = Model(inputs=[X, ini_acti_post, ini_mem_post], output=outputs)
    
    return model



# ## Predictions
# Once the model has been trained it is time to check its performance on new data and see how well it performs.
# for preprocessing date
def preprocess_date(date, human_char_idx):
    # truncate the length of date if it exceeds Tx
    if len(date) > Tx:
        date = date[:Tx]

    # some preprocessing
    date = date.lower().replace(',', '')
    source = np.zeros((1, Tx, len(human_char_idx)))

    # make OHE of date
    for t, char in enumerate(date):
        source[0, t, human_char_idx[char]] = 1

    return source


def user_interface(model):
    ch = 'y'
    while ch == 'y' or ch == 'Y':
        # clear the screen
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Enter input date (max string length 30) like 'sunday 15 september 2017', '29-oct-1996'")
        date = input()
        source = preprocess_date(date, human_char_idx)

        prediction = model.predict([source, ini_acti_post, ini_mem_post])
        prediction = np.argmax(prediction, axis=-1)
        output = [machine_idx_char[int(i)] for i in prediction]

        print("Output: ", ''.join(output))
        print()

        print('Continue ? Y or N')
        ch = input()
        

def main():

    # create model
    model = create_model(len(human_char_idx), len(machine_char_idx),
                        Tx, Ty, n_acti_pre, n_acti_post)
    model.summary()

    # load weights from any previously saved model
    model_path = r'models/weights_55.h5'

    if os.path.exists(model_path):
        model.load_weights(model_path)
        user_interface(model)
    else:
        print('First Train the model !!')




if __name__ == "__main__":
    main()
