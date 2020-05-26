import collections

import torch
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def load_vocab(vocab_file_path, sos, eos, pad, unk):
    """ Loads vocabulary from vocabulary file (create by vocabulary.py)"""
    vocab_file = open(vocab_file_path, "r+", encoding='utf-8')
    lines = vocab_file.readlines()

    vocab = collections.OrderedDict()
    # First, add special signs for sos, eos and pad to vocabulary
    vocab[pad] = PAD_ID
    vocab[sos] = SOS_ID
    vocab[eos] = EOS_ID
    vocab[unk] = UNK_ID
    # For each valid line, Get token and index of line
    for index, line in enumerate(lines):
        if line != "\n":
            token, count = line.replace("\n", "").split("\t")
            vocab[token] = index + 4  # first two values of vocabulary are taken

    return vocab


def convert_by_vocab(vocab, items, unk_val):
    """Converts a sequence of tokens or ids using the given vocabulary.
        If token is not in vocabulary, unk_val is return as default. """
    output = []
    for item in items:
        output.append(vocab.get(item, unk_val))
    return output


def tokenize_words(word_list):
    """Split words to lists of characters"""
    return [list(words) for words in word_list]


def tokenize_features(features_list):
    """Splits features by the separator sign ";" """
    return [connected_features.split(";") for connected_features in features_list]


class Tokenizer(object):
    """ Tokenizer object. Handles tokenizing sentences, converting tokens to ids and vice versa"""

    def __init__(self, src_vocab_file_path, tgt_vocab_file_path, device):
        self.sos = "<s>"
        self.eos = "</s>"
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.pad_id = PAD_ID
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

        self.device = device

        self.src_vocab = load_vocab(src_vocab_file_path, self.sos,
                                      self.eos, self.pad, self.unk)  # vocabulary of all token->id in the input
        self.inv_src_vocab = {v: k for k, v in self.src_vocab.items()}  # reverse vocabulary of input, id->token
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab = load_vocab(tgt_vocab_file_path, self.sos,
                                       self.eos, self.pad, self.unk)  # vocabulary of all token->id in the output
        self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}  # reverse vocabulary of output, id->token
        self.tgt_vocab_size = len(self.tgt_vocab)

        self.src_to_tgt_vocab_conversion_matrix = self.get_src_to_tgt_vocab_conversion_matrix()

    def add_sequence_symbols(self, tokens_list):
        """ Adds eos and sos symbols to each sequence of tokens"""
        return [[self.sos] + tokens + [self.eos] for tokens in tokens_list]

    def convert_src_tokens_to_ids(self, tokens):
        """ Converts all given tokens to ids using the input vocabulary"""
        return convert_by_vocab(self.src_vocab, tokens, self.unk_id)

    def convert_src_ids_to_tokens(self, ids):
        """ Converts all given ids to tokens using the input vocabulary"""
        return convert_by_vocab(self.inv_src_vocab, ids, self.unk)

    def convert_tgt_tokens_to_ids(self, tokens):
        """ Converts all given tokens to the ids using the output vocabulary"""
        return convert_by_vocab(self.tgt_vocab, tokens, self.unk_id)

    def convert_tgt_ids_to_tokens(self, ids):
        """ Converts all given tokens to the ids using the output vocabulary"""
        return convert_by_vocab(self.inv_tgt_vocab, ids, self.unk)

    def get_id_tensors(self, tokens_list, vocab_type):
        """ Gets list of token sequences, and converts each token sequence to tensor of ids, using the tokenizer
            device to determine tensor device type, and vocab type is either "INPUT" or "OUTPUT" """
        if vocab_type == "INPUT":
            return [torch.tensor(self.convert_src_tokens_to_ids(tokens), dtype=torch.long, device=self.device)
                    for tokens in tokens_list]
        else:
            return [torch.tensor(self.convert_tgt_tokens_to_ids(tokens), dtype=torch.long, device=self.device)
                    for tokens in tokens_list]

    def pad_tokens_sequence(self, tokens, max_seq_len):
        """ Pads the token sequence with pad symbols until it reaches the max sequence length.
            If Sequence is already at max length, nothing is added. """
        padding_len = max_seq_len - len(tokens)
        padding = [self.pad] * padding_len
        return tokens + padding

    def get_src_to_tgt_vocab_conversion_matrix(self):
        # Initialize conversion matrix
        src_to_tgt_conversion_matrix = torch.zeros(self.src_vocab_size, self.tgt_vocab_size, device=self.device)
        src_vocab_items = self.src_vocab.items()
        # Go over all (token, id) items in src vocab
        for src_token, src_id in src_vocab_items:
            tgt_id = self.tgt_vocab.get(src_token, self.unk_id)
            src_to_tgt_conversion_matrix[src_id][tgt_id] = 1
        return src_to_tgt_conversion_matrix

