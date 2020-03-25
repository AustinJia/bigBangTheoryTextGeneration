import os
import pickle

def load_data(path='data.pkl'):
    """
    Load Dataset from File
    """

    with open(path, 'rb') as data_file:
        data_dict = pickle.load(data_file)

    return data_dict


def create_lookup_tables(data_dict, token_dict={}):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    vocab = set()
    vocab_to_int, int_to_vocab = {}, {}

    # get all unique words in dataset
    for episode in data_dict:
        for line in data_dict[episode]:

            line = line.lower()

            # replace punctuations with special tokens
            for key, token in token_dict.items():
                text = line.replace(key, ' {} '.format(token))


            for word in line.split():
                vocab.add(word.strip())

    # create dicts
    for i, word in enumerate(vocab):
        vocab_to_int[word] = i
        int_to_vocab[i] = word

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """

    token_dict = {}
    dict_token[':'] = '||COLON||'
    token_dict[','] = '||COMMA||'
    token_dict['-'] = '||DASH||'
    token_dict['!'] = '||EXCLAMATION_MARK||'
    token_dict['?'] = '||QUESTION_MARK||'
    token_dict['"'] = '||QUOTATION_MARK||'
    token_dict['.'] = '||PERIOD||'
    token_dict[';'] = '||SEMICOLON||'
    token_dict['('] = '||LEFT_PARENTHESIS||'
    token_dict[')'] = '||RIGHT_PARENTHESIS||'

    return dict_token


def preprocess_and_save_data(dataset_path):
    """
    Preprocess Text Data
    """
    data_dict = load_data(dataset_path)
    token_dict = token_lookup()
    vocab_to_int, int_to_vocab = create_lookup_tables(text, token_dict)
    pickle.dump((vocab_to_int, int_to_vocab, token_dict), open('preprocess.pkl', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.pkl', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))
