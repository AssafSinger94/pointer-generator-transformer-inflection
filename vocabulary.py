import os
import itertools
import collections
import argparse

import data
import tokenizer
import utils

"""Reads conll file using functions in data (only train files are used to create a vocabulary). Using generic 
tokenizer functions (and not Tokenizer object that uses a vocabulary), creates input tokens vocabulary and target 
tokens vocabulary in different files. """


# Arguments
parser = argparse.ArgumentParser(
    description='Reads conll file (which is the dataset), and creates vocabulary files for input and output')
parser.add_argument('--src', type=str, default='train',
                    help="Source file of the dataset used to create the vocabulary (must include folder path)")
parser.add_argument('--vocab', type=str, default='vocab',
                    help="Target path of the vocabulary (must include folder path)")
args = parser.parse_args()

""" CONSTANTS """
WORD_FLAG = "WORD"
FEATURE_FLAG = "FEATURE"


def get_tokens_from_list(words_list, flag):
    """ Gets list of of either words or concatenated features, and returns one list of all tokens"""
    tokens_list = []
    if flag == WORD_FLAG:
        # Split words to lists of characters
        tokens_list = tokenizer.tokenize_words(words_list)
    else:
        # Split features by separator sign ";"
        tokens_list = tokenizer.tokenize_features(words_list)
    # Flat lists of tokens into one list of all tokens
    return list(itertools.chain.from_iterable(tokens_list))


def write_vocab_to_file(tokens_list, vocab_file_path):
    """
    Counts all tokens in list and writes them to file. Make dir if not exists.
    """
    utils.maybe_mkdir(vocab_file_path)
    vocab_file = open(vocab_file_path, "w", encoding='utf-8')  # "ISO-8859-1")
    # Get counter object to hold counts of characters
    vocab_counter = collections.Counter(tokens_list)
    # Write vocabulary (counter object) to file in order of frequency
    for vocab, count in vocab_counter.most_common():
        vocab_file.write(f"{vocab}\t{count}\n")

    vocab_file.close()


def create_vocab_files(src_file_path, vocab_file_path):
    """ Reads morph file and creates input tokens vocabulary and target tokens vocabulary, and writes them in
        different files """
    lemmas, targets, features = data.read_train_file(src_file_path)
    # Get tokens lists for source, target lemmas and features
    lemmas_tokens = get_tokens_from_list(lemmas, WORD_FLAG)
    targets_tokens = get_tokens_from_list(targets, WORD_FLAG)
    features_tokens = get_tokens_from_list(features, FEATURE_FLAG)
    # input tokens = lemmas tokens + features tokens
    input_tokens = lemmas_tokens + targets_tokens + features_tokens
    output_tokens = lemmas_tokens + targets_tokens
    # write vocabularies of inputs and outputs to files
    write_vocab_to_file(input_tokens, vocab_file_path + "-input")
    write_vocab_to_file(output_tokens, vocab_file_path + "-output")


if __name__ == '__main__':
    # Create vocab files
    create_vocab_files(args.src, args.vocab)
