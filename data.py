import re
import tokenizer

"""Allows reading of conll files. Reads conll file, and 1) Splits it by different components of each examples, 
and also 2) Separates it to to input tokens and target tokens. conll: lemma-tab-target-tab-features """


""" READING FILES """
def read_morph_file(morph_file_path):
    """ Reads conll file, split to line, and splits each line by tabs. Returns list of lists"""
    # Get all lines in file
    morph_file = open(morph_file_path, 'r', encoding='utf-8')
    lines = morph_file.readlines()
    outputs = []
    # Separate lines to proper format
    for line in lines:
        if line != "\n":
            # Strip '\n' and split
            outputs.append(line.replace("\n", "").split("\t"))
    morph_file.close()
    return outputs


def clean_word(word):
    """ Strips word of unnecessary symbols """
    word = re.sub("[!@#$']", '', word)
    return word.lower()


def read_train_file(train_file_path):
    """ Reads conll train file, and splits to lists of input lemmas, input features and target lemmas"""
    lemmas = []
    targets = []
    features = []
    train_morph_list = read_morph_file(train_file_path)

    for lemma, target, feature in train_morph_list:
        # Add results to relevant lists
        lemmas.append(lemma)
        features.append(feature)
        targets.append(target)

    return lemmas, targets, features


def read_test_file(test_input_file):
    """ Reads conll test file, and splits to lists of input lemmas, input features and target lemmas"""
    lemmas = []
    features = []
    test_morph_list = read_morph_file(test_input_file)

    for lemma, feature in test_morph_list:
        # Add results to relevant lists
        lemmas.append(lemma)
        features.append(feature)

    return lemmas, features


def read_train_file_tokens(train_file_path):
    """ Reads conll train file, and splits to input tokens and target tokens.
        Each input and target is a list of tokens"""
    lemmas, targets, features = read_train_file(train_file_path)
    # tokenize all three lists, get as list of tokens lists
    lemmas_tokens = tokenizer.tokenize_words(lemmas)
    targets_tokens = tokenizer.tokenize_words(targets)
    features_tokens = tokenizer.tokenize_features(features)
    # concatenate feature tokens to lemma tokens
    input_tokens = [lemma_tokens + feature_tokens for lemma_tokens, feature_tokens in
                    zip(lemmas_tokens, features_tokens)]
    return input_tokens, targets_tokens


def read_test_file_tokens(test_file_path):
    """ Reads conll test file. Each input is a list of tokens"""
    lemmas, features = read_test_file(test_file_path)
    # tokenize all two lists, get as list of tokens lists
    lemmas_tokens = tokenizer.tokenize_words(lemmas)
    features_tokens = tokenizer.tokenize_features(features)
    # concatenate feature tokens to lemma tokens
    input_tokens = [lemma_tokens + feature_tokens for lemma_tokens, feature_tokens in
                    zip(lemmas_tokens, features_tokens)]
    return input_tokens


""" WRITING FILES """
def write_morph_file(lemmas, targets, features, out_file_path):
    """ Writes tokens as conll file """
    # Get all lines in file
    out_file = open(out_file_path, 'w', encoding='utf-8')
    for lemma, target, feature in zip(lemmas, targets, features):
        out_file.write("%s\t%s\t%s\n" % (lemma, target, feature))
    out_file.close()
