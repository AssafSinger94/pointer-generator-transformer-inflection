import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import dataset
import data
import tokenizer

# Arguments
import transformer
import utils

parser = argparse.ArgumentParser(description='Evaluating the transformer over test and validation sets')
parser.add_argument('--model-checkpoint', type=str, default='checkpoints/model_best.pth',
                    help="the model file to be evaluated. Usually is of the form model_X.pth (must include folder path)")
parser.add_argument('--arch', type=str, default='transformer',
                    help="Architecture type for model: transformer, pointer_generator")
parser.add_argument('--embed-dim', type=int, default=128,
                    help='Embedding dimension (default: 128)')
parser.add_argument('--fcn-dim', type=int, default=256,
                    help='Fully-connected network hidden dimension (default: 256)')
parser.add_argument('--num-heads', type=int, default=4,
                    help='number of attention heads (default: 4)')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of layers in encoder and decoder (default: 2)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--test', type=str, default='data',
                    help="Test file of the dataset (must include folder path)")
parser.add_argument('--vocab', type=str, default='data',
                    help="Base name of vocabulary files (must include folder path)")
parser.add_argument('--pred', type=str, default='pred',
                    help="Name of output file containing predictions of the test set (must include folder path)")
args = parser.parse_args()

""" FILES AND TOKENIZER """
# Get test and out file path
test_file = args.test
out_file = args.pred
# Get vocabulary paths
src_vocab_file = args.vocab + "-input"
tgt_vocab_file = args.vocab + "-output"

# Log all relevant files
logger = utils.get_logger()
logger.info(f"Model checkpoint: {args.model_checkpoint}")
logger.info(f"Test file: {test_file}")
logger.info(f"Input vocabulary file: {src_vocab_file}")
logger.info(f"Output vocabulary file: {tgt_vocab_file}")
logger.info(f"Prediction file: {out_file}")

""" CONSTANTS """
MAX_SRC_SEQ_LEN = 45
MAX_TGT_SEQ_LEN = 45

""" MODEL AND DATA LOADER """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize Tokenizer object with input and output vocabulary files
myTokenizer = tokenizer.Tokenizer(src_vocab_file, tgt_vocab_file, device=device)
# Load model from checkpoint in evaluation mode
model = utils.build_model(args.arch, myTokenizer.src_vocab_size, myTokenizer.tgt_vocab_size, args.embed_dim, args.fcn_dim,
                args.num_heads, args.num_layers, args.dropout, myTokenizer.src_to_tgt_vocab_conversion_matrix)
model = utils.load_model(model, args.model_checkpoint, logger)
model.to(device)
model.eval()
# Initialize DataLoader object
data_loader = dataset.DataLoader(myTokenizer, train_file_path=None, valid_file_path=None,
                                 test_file_path=test_file, device=device)

""" FUNCTIONS """


def prdeict_word(src, max_seq_len):
    # Add batch dimension
    src = src.unsqueeze(dim=0)
    src_key_padding_mask = data_loader.get_padding_mask(src)
    memory = model.encode(src, src_key_padding_mask)
    outputs = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
    outputs[0] = myTokenizer.sos_id
    for j in range(1, max_seq_len):
        # Compute output of model
        tgt_key_padding_mask = data_loader.get_padding_mask(outputs[:, :j])
        out = model.decode(memory, outputs[:, :j], tgt_key_padding_mask, src_key_padding_mask).squeeze() if \
            (model.__class__.__name__ == "Transformer") else \
            model.decode(memory, outputs[:, :j], src, tgt_key_padding_mask, src_key_padding_mask).squeeze()
        val, ix = out.topk(1)
        outputs[0, j] = ix[-1]
        if ix[-1] == myTokenizer.eos_id:
            break
    # Strip off sos and eos tokens
    return outputs[0, 1:j]


# --------For Transformer model from SIGMORPHON 2020 Baseline----------
def dummy_mask(seq):
    '''
    create dummy mask (all 1)
    '''
    if isinstance(seq, tuple):
        seq = seq[0]
    assert len(seq.size()) == 1 or (len(seq.size()) == 2 and seq.size(1) == 1)
    return torch.ones_like(seq, dtype=torch.float)


def decode_greedy_transformer(src_sentence, max_len=40, trg_bos=myTokenizer.sos_id, trg_eos=myTokenizer.eos_id):
    '''
    src_sentence: [seq_len, 1]
    '''
    model.eval()
    src_mask = dummy_mask(src_sentence)
    src_mask = (src_mask == 0).transpose(0, 1)
    enc_hs = model.encode(src_sentence, src_mask)

    output, attns = [trg_bos], []

    for _ in range(max_len):
        output_tensor = torch.tensor(output, device=device).view(len(output), 1)
        trg_mask = dummy_mask(output_tensor)
        trg_mask = (trg_mask == 0).transpose(0, 1)

        word_logprob = model.decode(enc_hs, src_mask, output_tensor, trg_mask)
        word_logprob = word_logprob[-1]

        word = torch.max(word_logprob, dim=1)[1]
        if word == trg_eos:
            break
        output.append(word.item())
    return output[1:]  # , attns
# ------------------------------------


def write_predictions_to_file(predictions, test_file_path, out_file_path):
    utils.maybe_mkdir(out_file_path)
    # Get original input from test file
    lemmas, features = data.read_test_file(test_file_path)
    # Write all data with predictions to the out file
    data.write_morph_file(lemmas, predictions, features, out_file_path)


def generate_prediction_file(max_seq_len=MAX_TGT_SEQ_LEN):
    """ Generates predictions over the test set and prints output to prediction file."""
    input_ids, input_tokens = data_loader.get_test_set()
    predictions = []
    # Go over each example
    for i, (data, data_tokens) in tqdm(enumerate(zip(input_ids, input_tokens))):
        unkown_tokens = [token for token in data_tokens if token not in myTokenizer.src_vocab]
        # Get prediction from model
        # ------------------
        pred = prdeict_word(data, max_seq_len)
        # Convert from predicted ids to the predicted word
        pred_tokens = myTokenizer.convert_tgt_ids_to_tokens(pred.tolist())
        # pred = decode_greedy_transformer(data.unsqueeze(dim=0).transpose(0, 1), max_seq_len)
        # pred_tokens = myTokenizer.convert_tgt_ids_to_tokens(list(pred))
        # ------------------

        # where token is unkown token, copy from the source at the same token location
        unkown_idx = 0
        for j in range(len(pred_tokens)):
            if pred_tokens[j] == myTokenizer.unk and (j < len(data_tokens) - 1):
                pred_tokens[j] = data_tokens[j + 1]  # account for data token padded with <s> at the beginning
            # if pred_tokens[j] == myTokenizer.unk:
            #     pred_tokens[j] = unkown_tokens[unkown_idx]
            #     # Increment index, until reaches the end, then stay
            #     unkown_tokens = min(unkown_tokens + 1, len(unkown_tokens) - 1)

        pred_word = ''.join(pred_tokens)
        predictions.append(pred_word)
    write_predictions_to_file(predictions, test_file, out_file)


if __name__ == '__main__':
    # Generate predictions for test set
    generate_prediction_file()
    logger.info(f"Created prediction file: {out_file}\n")


# def prdeict_word(src, max_seq_len):
#     """
#      Predicts target word given source (lemma + features). Predictions generated in greedy manner.
#     """
#     # Add batch dimension
#     src = src.unsqueeze(dim=0)
#     src = src.transpose(0, 1)
#     outputs = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
#     outputs[0] = myTokenizer.sos_id
#     for j in range(1, max_seq_len):
#         trg = outputs[:, :j]
#         trg = trg.transpose(0, 1)
#         src_mask = (src > 0).float()
#         trg_mask = (trg > 0).float()
#         # Compute output of model
#         out = model(src, src_mask, trg, trg_mask).transpose(0, 1).squeeze()
#         val, ix = out.topk(1)
#         outputs[0, j] = ix[-1]
#         if ix[-1] == myTokenizer.eos_id:
#             break
#     return outputs[0, :j + 1]
#
