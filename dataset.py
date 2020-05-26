from random import shuffle
import torch

import data
# import tokenizer


def get_train_dataset(train_file_path, tokenizer, max_src_seq_len=30, max_tgt_seq_len=25):
    """
    Reads input and output tokens from train set file, and converts tokens to tensors of ids using tokenizer.
    """
    # Read input and output tokens from dataset
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(train_file_path)
    # Pad with sos and eos
    inputs_tokens = tokenizer.add_sequence_symbols(inputs_tokens)
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Split target into two targets, for teacher forcing
    # -------------
    targets_tokens = [target_tokens[:-1] for target_tokens in outputs_tokens]
    # --------For Transformer model from SIGMORPHON 2020 Baseline----------
    # targets_tokens = outputs_tokens
    # -------------
    targets_y_tokens = [target_tokens[1:] for target_tokens in outputs_tokens]

    # Get lists of all input ids, target ids and target_y ids, where each sequence padded up to max length
    inputs_ids = [
        tokenizer.convert_src_tokens_to_ids(tokenizer.pad_tokens_sequence(input_tokens, max_src_seq_len))
        for input_tokens in inputs_tokens]
    targets_ids = [
        tokenizer.convert_tgt_tokens_to_ids(tokenizer.pad_tokens_sequence(target_tokens, max_tgt_seq_len))
        for target_tokens in targets_tokens]
    targets_y_ids = [
        tokenizer.convert_tgt_tokens_to_ids(tokenizer.pad_tokens_sequence(target_y_tokens, max_tgt_seq_len))
        for target_y_tokens in targets_y_tokens]

    return inputs_ids, targets_ids, targets_y_ids


def get_valid_dataset(valid_file_path, tokenizer):
    """
    Reads input and output tokens from valid set file, and converts tokens to tensors of ids using tokenizer.
    """
    # Read input and output tokens from dataset
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(valid_file_path)
    # Pad with sos and eos
    inputs_tokens = tokenizer.add_sequence_symbols(inputs_tokens)
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Get tensors of all input ids and output ids
    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, "INPUT")
    outputs_ids = tokenizer.get_id_tensors(outputs_tokens, "OUTPUT")
    return inputs_ids, outputs_ids


def get_test_dataset(test_file_path, tokenizer):
    """
    Reads tokens from test set file, and converts tokens to tensors of ids using tokenizer.
    Returns the tokens as well, used for prediction.
    """
    # Read input tokens from dataset
    inputs_tokens = data.read_test_file_tokens(test_file_path)
    # Pad with sos and eos
    inputs_tokens = tokenizer.add_sequence_symbols(inputs_tokens)
    # Get tensors of input ids
    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, "INPUT")
    return inputs_ids, inputs_tokens

def shuffle_together(list1, list2, list3):
    """Shuffles two lists together"""
    zip_list = list(zip(list1, list2, list3))
    shuffle(zip_list)
    list1, list2, list3 = zip(*zip_list)
    return list1, list2, list3

def split_to_batches(ids_list, device, batch_size=128):
    """ splits list of id sequence into batchs.
        Gets list of sequences (list of size seq_len)
        returns list of batchs, each batch is a tensor of size N x S (batch_size x seq_len)"""
    return [torch.tensor(ids_list[x:x + batch_size], dtype=torch.long, device=device) for x in
            range(0, len(ids_list), batch_size)]


def get_batches(input_ids, target_ids, target_y_ids, device, batch_size=128):
    """ Gets entire dataset, shuffles the data, and splits it to batches.
        Each batch is a tensor of size N x S (batch_size x seq_len)."""
    # Shuffle together
    shuffled_input_ids, shuffled_target_ids, shuffled_target_y_ids = shuffle_together(input_ids, target_ids, target_y_ids)
    # split to batches
    input_ids_batches = split_to_batches(shuffled_input_ids, device, batch_size)
    target_ids_batches = split_to_batches(shuffled_target_ids, device, batch_size)
    target_y_ids_batches = split_to_batches(shuffled_target_y_ids, device, batch_size)
    return input_ids_batches, target_ids_batches, target_y_ids_batches


class DataLoader(object):
    """ Contains all utilities for reading train/valid/test sets """
    def __init__(self, tokenizer, train_file_path=None, valid_file_path=None, test_file_path=None,
                 device="cpu", batch_size=128, max_src_seq_len=30, max_tgt_seq_len=25):
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_src_seq_len = max_src_seq_len
        self.max_tgt_seq_len = max_tgt_seq_len
        # Read train file and get train set
        if train_file_path is not None:
            train_input_ids, train_target_ids, train_target_y_ids = get_train_dataset(train_file_path, tokenizer,
                                                                                      self.max_src_seq_len, self.max_tgt_seq_len)
            self.train_input_ids = train_input_ids
            self.train_target_ids = train_target_ids
            self.train_target_y_ids = train_target_y_ids
            self.train_set_size = len(self.train_input_ids)
        else:
            self.train_input_ids = None
            self.train_target_ids = None
            self.train_target_y_ids = None
            self.train_set_size = 0

        if valid_file_path is not None:
            # Read validation file and get validation set, for checking loss using teacher forcing
            valid_input_ids_tf, valid_target_ids_tf, valid_target_y_ids_tf = get_train_dataset(valid_file_path, tokenizer,
                                                                                               self.max_src_seq_len, self.max_tgt_seq_len)
            self.valid_input_ids_tf = valid_input_ids_tf
            self.valid_target_ids_tf = valid_target_ids_tf
            self.valid_target_y_ids_tf = valid_target_y_ids_tf
            # Read validation file and get validation set, for evaluation
            valid_input_ids, valid_target_ids = get_valid_dataset(valid_file_path, tokenizer)
            self.valid_input_ids = valid_input_ids
            self.valid_target_ids = valid_target_ids
        else:
            self.valid_input_ids_tf = None
            self.valid_target_ids_tf = None
            self.valid_target_y_ids_tf = None
            self.valid_input_ids = None
            self.valid_target_ids = None

        if test_file_path is not None:
            # Read test file and get test set
            test_input_ids = get_test_dataset(test_file_path, tokenizer)
            self.test_input_ids = test_input_ids
        else:
            self.test_input_ids = None

    def get_train_set(self):
        return get_batches(self.train_input_ids, self.train_target_ids, self.train_target_y_ids,
                           self.device, batch_size=self.batch_size)

    def get_validation_set_tf(self):
        return get_batches(self.valid_input_ids_tf, self.valid_target_ids_tf, self.valid_target_y_ids_tf,
                           self.device, batch_size=self.batch_size)

    def get_validation_set(self):
        return self.valid_input_ids, self.valid_target_ids

    def get_validation_set_len(self):
        return len(self.valid_input_ids)

    def get_test_set(self):
        return self.test_input_ids

    def get_test_set_len(self):
        return len(self.test_input_ids)

    def get_padding_mask(self, batch_tensor):
        """" Returns padding masks for given batch
            Padding masks are ByteTensor where True values are positions that are masked and False values are not.
            inputs are of size N x S (batch_size x seq_len)
            Returns masks of same size - N x S (batch_size x seq_len) """
        return batch_tensor == self.tokenizer.pad_id

    def get_padding_masks(self, source_batch, target_batch):
        """" Returns padding masks for source batch, memory batch, and target batch. """
        src_padding_mask = self.get_padding_mask(source_batch)
        mem_padding_mask = src_padding_mask
        target_padding_mask = self.get_padding_mask(target_batch)
        return src_padding_mask, mem_padding_mask, target_padding_mask
