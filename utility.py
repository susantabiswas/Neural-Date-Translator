import numpy as np
import random
from faker import Faker
from babel.dates import format_date

# for creating a single fake date


def create_date(fake_obj, DATE_FORMATS):

    try:
        # create a date object
        dt = fake_obj.date_object()
        # create machine readable dates
        machine_read_dates = dt.isoformat()
        # get human readable dates
        human_read_dates = format_date(dt, format=random.choice(DATE_FORMATS),
                                       locale='en_IN')
        # remove punctuations
        human_read_dates = human_read_dates.replace(',', '')
        # change to lower case
        human_read_dates = human_read_dates.lower()

    except AttributeError as e:
        return None, None

    return human_read_dates, machine_read_dates


# for creating 'm' training dataset
def create_dataset(m):
    '''
    Arg:
        m: no. of training examples
    Returns:
    dataset --list: for saving list of tuples of date pairs
    human_vocab --set: for saving human readable dates vocabulary
    machine_vocab -- set: for saving machine readable dates vocabulary
    
    '''
    # for generating fake training data
    fake_obj = Faker()

    # date formats for generating date
    # one of them is selected randomly each time that is why full is mentioned
    # many times to increase its chances
    DATE_FORMATS = ['short',
                    'medium',
                    'long',
                    'full',
                    'full',
                    'full',
                    'full',
                    'full',
                    'full',
                    'full',
                    'full',
                    'full',
                    'full',
                    'd MMM YYY',
                    'd MMMM YYY',
                    'dd MMM YYY',
                    'd MMM, YYY',
                    'd MMMM, YYY',
                    'dd, MMM YYY',
                    'd MM YY',
                    'd MMMM YYY',
                    'MMMM d YYY',
                    'MMMM d, YYY',
                    'dd.MM.YY']
    # for saving the dataset
    dataset = []
    # for saving human readable dates vocabulary
    human_vocab = set()
    # for saving machine readable dates vocabulary
    machine_vocab = set()

    for i in range(m):
        human_date, machine_date = create_date(fake_obj, DATE_FORMATS)
        if human_date is not None:
            # add date to dataset
            dataset.append((human_date, machine_date))
            # add new vocabulary entry
            for char in human_date:
                if char not in human_vocab:
                    human_vocab.add(char)
            for char in machine_date:
                if char not in machine_vocab:
                    machine_vocab.add(char)

    human_vocab = sorted(human_vocab)
    machine_vocab = sorted(machine_vocab)

    return dataset, human_vocab, machine_vocab
    

def preprocess_data(m, dataset, human_char_idx, machine_char_idx, Tx, Ty):
    # separate the tuples
    X, Y = zip(*dataset)

    # make numpy arrays to store the X and Y data
    X_ohe = np.zeros((m, Tx, len(human_char_idx)), dtype='float32')
    Y_ohe = np.zeros((m, Ty, len(machine_char_idx)), dtype='float32')

    # truncate the length of date if it exceeds Tx
    for i, date in enumerate(X):
        if len(date) > Tx:
            X[i] = X[:Tx]

    # now do one hot encoding
    for i in range(m):
        for timestep, char in enumerate(X[i]):
            X_ohe[i, timestep, human_char_idx[char]] = 1
        for timestep, char in enumerate(Y[i]):
            Y_ohe[i, timestep, machine_char_idx[char]] = 1

    return X, Y, X_ohe, Y_ohe


def create_training_data(m, Tx, Ty):
    dataset, human_vocab, machine_vocab = create_dataset(m)
    
    # now we will create a dictionary for mapping the vocabulary tokens to numerical indices
    human_char_idx = dict((token, i) for i, token in enumerate(human_vocab))
    # reverse mapping from indices to tokens for machine readable dates
    machine_idx_char = dict(enumerate(machine_vocab))
    # mapping from char to indices for machine readable dates
    machine_char_idx = dict((token, i)
                            for i, token in enumerate(machine_vocab))

    X, Y, X_ohe, Y_ohe = preprocess_data(
        m, dataset, human_char_idx, machine_char_idx, Tx, Ty)

    return dataset, X, Y, X_ohe, Y_ohe, human_vocab, human_char_idx, machine_vocab, machine_char_idx, machine_idx_char
