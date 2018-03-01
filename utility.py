import numpy as np
import random
from faker import Faker
from babel.dates import format_date
from keras.utils import to_categorical


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
    DATE_FORMATS = [
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
        'dd.MM.YY',
        'short',
        'full'
        'medium',
        'long',
        'full'
        'full'
        'full']
    # for saving the dataset
    dataset = []
    # for saving human readable dates vocabulary
    human_vocab = set()
    # for saving machine readable dates vocabulary
    machine_vocab = set()

    for i in range(m):
        human, machine = create_date(fake_obj, DATE_FORMATS)
        if human is not None:
            # add date
            dataset.append((human, machine))
        # add vocabulary entry
        if human not in human_vocab:
            human_vocab.add(human)
        if machine not in machine_vocab:
            machine_vocab.add(machine)

    return dataset, sorted(human_vocab), sorted(machine_vocab)


# for preprocessing the data
def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    # separate the tuples
    X, Y = zip(*dataset)
   
    # change the string dates to list of indices representing the string dates
    X = np.array([date_to_indices(date, Tx, human_vocab) for date in X])
    Y = np.array([date_to_indices(date, Ty, machine_vocab) for date in Y])
    
    # one hot encoding of training input and output
    X_ohe = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Y_ohe = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))
    
    return X, Y, X_ohe, Y_ohe

# for converting a date to numerical valued indices
def date_to_indices(date, Tx, vocab):
    # make the string lower and remove ','
    date = date.lower()
    date = date.replace(',', '')

    # if the string length is greater than Tx then it needs to be cut
    if len(date) > Tx:
        date = date[:,Tx]
    
    new_date = list(map(lambda x: vocab.get(x, '<UNK>'), date))

    # if the length is smaller fill the remaining time steps with padding
    if len(date) < Tx:
        new_date += [vocab['<PAD>']] * (Tx - len(date))

    return new_date


# for converting from index to char from machine_vocab
def indices_to_date(indices, machine_idx_char):
    """
    Arguments:
        indices -- list of integers representing indexes in the machine's vocabulary
        machine_idx_char -- dict mapping machine readable indices to machine readable characters 
    
    Returns:
        date -- list of characters representing machine readable date 
    """

    date = [machine_idx_char[i] for i in indices]
    return date


# for creating the training examples in the right format
def create_training_data(m, Tx, Ty):
    dataset, human_vocab, machine_vocab = create_dataset(m)
    # add the unknown and pad characters
    human_vocab += ['<UNK>', '<PAD>']
    # now we will create a dictionary for mapping the vocabulary tokens to numerical indices
    human_char_idx = dict((token, i) for i, token in enumerate(human_vocab) )
    # reverse mapping from indices to tokens for machine readable dates
    machine_idx_char = dict(enumerate(machine_vocab))
    # mapping from char to indices for machine readable dates
    machine_char_idx = dict((token, i) for i, token in enumerate(machine_vocab) )

    X, Y, X_ohe, Y_ohe = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

    return dataset, X, Y, X_ohe, Y_ohe, human_vocab, machine_vocab, machine_char_idx, machine_idx_char    
