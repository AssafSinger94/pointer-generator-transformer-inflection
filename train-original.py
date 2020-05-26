import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
import dataset
import tokenizer

import transformer_baseline

# Training parameters
parser = argparse.ArgumentParser(description='Training Transformer and Pointer-Generator for morphological inflection')
parser.add_argument('--train', type=str, default='data',
                    help="Train file of the dataset (File is located in DATA_FOLDER)")
parser.add_argument('--dev', type=str, default='data',
                    help="Validation file of the dataset (File is located in DATA_FOLDER)")
parser.add_argument('--vocab', type=str, default='data',
                    help="Base name of vocabulary files (must include dir path)")
parser.add_argument('--checkpoints-dir', type=str, default='model-checkpoints',
                    help='Folder to keep checkpoints of model')
parser.add_argument('--resume', type=bool, default=False,
                    help="Whether to resume training from a certain checkpoint")
parser.add_argument('--reload', type=bool, default=False,
                    help="Whether to reload pretrained model from certain checkpoint")
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--steps', type=int, default=100,
                    help='number of batch steps to train (default: 20,000)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--eval-every', type=int, default=1,
                    help='Evaluate model over validation set every how many epochs (default: 1)')
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
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate (default: 0.01)')
parser.add_argument('--beta', type=float, default=0.9,
                    help='beta for Adam optimizer (default: 0.01)')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta 2 for Adam optimizer (default: 0.01)')
parser.add_argument('--label-smooth', default=0.1, type=float,
                    help='label smoothing coeff')
parser.add_argument('--scheduler', type=str, default="ReduceLROnPlateau",
                    help='Learning rate Scheduler (default: ReduceLROnPlateau)')
parser.add_argument('--patience', default=5, type=int,
                    help='patience of for early stopping (default: 0)')
parser.add_argument('--min-lr', type=float, default=1e-5,
                    help='Minimum learning rate (default: 0.01)')
parser.add_argument('--discount-factor', default=0.5, type=float,
                    help='discount factor of `ReduceLROnPlateau` (default: 0.5)')
parser.add_argument('--patience_reduce', default=0, type=int,
                    help='patience of `ReduceLROnPlateau` (default: 0)')
parser.add_argument('--warmup-steps', default=4000, type=int,
                    help='number of warm up steps for scheduler (default: 4000)')
args = parser.parse_args()

# Get train and validation file paths
train_file = args.train
valid_file = args.dev
# Get vocabulary paths
src_vocab_file = args.vocab + "-input"
tgt_vocab_file = args.vocab + "-output"
# Initialize Tokenizer object with input and output vocabulary files
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myTokenizer = tokenizer.Tokenizer(src_vocab_file, tgt_vocab_file, device)

""" CONSTANTS """
MAX_SRC_SEQ_LEN = 45
MAX_TGT_SEQ_LEN = 45
SRC_VOCAB_SIZE = myTokenizer.src_vocab_size
TGT_VOCAB_SIZE = myTokenizer.tgt_vocab_size
# Model Hyperparameters
EMBEDDING_DIM = args.embed_dim
FCN_HIDDEN_DIM = args.fcn_dim
NUM_HEADS = args.num_heads
NUM_LAYERS = args.num_layers
DROPOUT = args.dropout
# BEST MODEL FOR MEDIUM RESOURCE
# EMBEDDING_DIM = 64
# FCN_HIDDEN_DIM = 256
# NUM_HEADS = 4
# NUM_LAYERS = 2
# DROPOUT = 0.2
# # BEST MODEL FOR LOW RESOURCE
# EMBEDDING_DIM = 128
# FCN_HIDDEN_DIM = 64
# NUM_HEADS = 4
# NUM_LAYERS = 2
# DROPOUT = 0.2

""" MODEL AND DATA LOADER """
model = utils.build_model(args.arch, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMBEDDING_DIM, FCN_HIDDEN_DIM,
                          NUM_HEADS, NUM_LAYERS, DROPOUT, myTokenizer.src_to_tgt_vocab_conversion_matrix)
# --------Transformer model from SIGMORPHON 2020 Baseline----------
# model = transformer_baseline.Transformer(src_vocab_size=SRC_VOCAB_SIZE, trg_vocab_size=TGT_VOCAB_SIZE,
#                                      embed_dim=EMBEDDING_DIM, nb_heads=NUM_HEADS,
#                                      src_hid_size=FCN_HIDDEN_DIM, src_nb_layers=NUM_LAYERS,
#                                      trg_hid_size=FCN_HIDDEN_DIM, trg_nb_layers=NUM_LAYERS,
#                                      dropout_p=DROPOUT,
#                                      tie_trg_embed=False, src_c2i=None, trg_c2i=None, attr_c2i=None, label_smooth=0.1)
model.to(device)
criterion = nn.NLLLoss(reduction='mean', ignore_index=myTokenizer.pad_id)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta, args.beta2))
scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=args.min_lr, factor=args.discount_factor,
                              patience=args.patience_reduce) \
    if (args.scheduler == "ReduceLROnPlateau") \
    else utils.WarmupInverseSquareRootSchedule(optimizer, args.warmup_steps)

# Initialize DataLoader object
data_loader = dataset.DataLoader(myTokenizer, train_file_path=train_file, valid_file_path=valid_file,
                                 test_file_path=None, device=device, batch_size=args.batch_size,
                                 max_src_seq_len=MAX_SRC_SEQ_LEN, max_tgt_seq_len=MAX_TGT_SEQ_LEN)
# data_loader = dataset.DataLoader(myTokenizer, train_file_path=train_file, valid_file_path=None,
#                                  test_file_path=None, device=device, batch_size=args.batch_size,
#                                  max_src_seq_len=MAX_SRC_SEQ_LEN, max_tgt_seq_len=MAX_TGT_SEQ_LEN)


""" HELPER FUNCTIONS"""
def get_lr():
    if isinstance(scheduler, ReduceLROnPlateau):
        return optimizer.param_groups[0]['lr']
    try:
        return scheduler.get_last_lr()[0]
    except:
        return scheduler.get_lr()[0]


def get_loss(predict, target):
    """
    Compute loss
    :param predict: SxNxTGT_VOCAB
    :param target: SxN
    :return: loss
    """
    predict = predict.contiguous().view(-1, TGT_VOCAB_SIZE)
    # nll_loss = F.nll_loss(predict, target.view(-1), ignore_index=PAD_IDX)
    target = target.contiguous().view(-1, 1)
    non_pad_mask = target.ne(myTokenizer.pad_id)
    nll_loss = -predict.gather(dim=-1, index=target)[non_pad_mask].mean()
    smooth_loss = -predict.sum(dim=-1, keepdim=True)[non_pad_mask].mean()
    smooth_loss = smooth_loss / TGT_VOCAB_SIZE
    loss = (1. -
            args.label_smooth) * nll_loss + args.label_smooth * smooth_loss
    return loss


""" LOGGING SETTINGS AND LOGGING """
# Set number of total epoch and min epoch for eval start
MIN_EVAL_STEPS = 40
steps_per_epoch = int(math.ceil(data_loader.train_set_size / args.batch_size))
epochs = int(math.ceil(args.steps / steps_per_epoch))
min_eval_epochs = int(MIN_EVAL_STEPS / steps_per_epoch)

logger = utils.get_logger()
logger.info(f"Starting training. resume training: {args.resume}, reload from pretraining: {args.reload}")
logger.info(f"Arch: {args.arch}, embed_dim: {EMBEDDING_DIM}, fcn_hid_dim: {FCN_HIDDEN_DIM},"
            f" num-heads: {NUM_HEADS}, num-layers: {NUM_LAYERS}, dropout: {DROPOUT}, device: {device}")
logger.info(f"Steps: {args.steps}, batch size:{args.batch_size}, Steps per epoch {steps_per_epoch},\n"
            f" Epochs: {epochs}, Eval every :{args.eval_every}")
logger.info(f"Optimizer: Adam, lr: {args.lr}, beta: {args.beta}, beta2: {args.beta2}")
logger.info(
    f"Scheduler: {args.scheduler}, patience: {args.patience}, min_lr: {args.min_lr}, warmup steps: {args.warmup_steps},"
    f" discount factor: {args.discount_factor}, patience_reduce: {args.patience_reduce}")
logger.info(f"Source vocabulary: Size = {myTokenizer.src_vocab_size}, {myTokenizer.src_vocab}")
logger.info(f"Target vocabulary: Size = {myTokenizer.tgt_vocab_size}, {myTokenizer.tgt_vocab}")
logger.info(f"Training file: {train_file}")
logger.info(f"Validation file: {valid_file}")
logger.info(f"Input vocabulary file: {src_vocab_file}")
logger.info(f"Output vocabulary file: {tgt_vocab_file}")
logger.info(f"Checkpoints dir: {args.checkpoints_dir}")
logger.info(f"Model: {model}")

# Reload model/ resume training if applicable
if args.resume:
    # Resume training from checkpoint
    model, optimizer, scheduler, start_epoch, best_valid_accuracy = \
        utils.load_checkpoint(model, optimizer, scheduler, f"{args.checkpoints_dir}/model_best.pth", logger)
    best_valid_epoch = start_epoch
else:
    # Reload pretrained model from checkpoint
    if args.reload:
        model = utils.load_model(model, f"{args.checkpoints_dir}/model_best.pth", logger)
    start_epoch = 0
    # Initialize best validation loss placeholders
    best_valid_accuracy = -1.0
    best_valid_epoch = 0

""" FUNCTIONS """
def train(epoch):
    """ Runs full training epoch over the training set, uses teacher forcing in training"""
    model.train()
    running_loss = 0.0
    # Get Training set in batches
    input_ids_batches, target_ids_batches, target_y_ids_batches = data_loader.get_train_set()
    # Go over each batch
    for i, (data, target, target_y) in tqdm(enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches))):
        # for i, (data, target, target_y) in enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches)):
        optimizer.zero_grad()
        # Get padding masks
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target)
        # Compute output of model
        output = model(data, target, src_pad_mask, target_pad_mask, mem_pad_mask)
        # ---------------
        # Compute loss
        # loss = criterion(output.contiguous().view(-1, TGT_VOCAB_SIZE), target_y.contiguous().view(-1))
        loss = get_loss(output.transpose(0, 1), target_y.transpose(0, 1))
        # -------------
        # Propagate loss and update model parameters
        loss.backward()
        optimizer.step()
        if not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        running_loss += loss.item()
    # print statistics
    logger.info(f"Train Epoch: {epoch}, avg loss: {running_loss / (i + 1):.4f}, lr {get_lr():.6f}")


def validation(epoch):
    """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
    model.eval()
    running_loss = 0
    correct_preds = 0
    # Get Training set batches
    input_ids_batches, target_ids_batches, target_y_ids_batches = data_loader.get_validation_set_tf()
    # Go over each batch
    for i, (data, target, target_y) in enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches)):
        # Get padding masks
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target)
        # Compute output of model
        output = model(data, target, src_pad_mask, target_pad_mask, mem_pad_mask)
        # Get model predictions
        predictions = output.topk(1)[1].squeeze()
        # Compute accuracy
        target_pad_mask = (target_pad_mask == False).int()
        predictions = predictions * target_pad_mask
        correct_preds += torch.all(torch.eq(predictions, target_y), dim=-1).sum()
        # ---------------
        # Compute loss
        # loss = criterion(output.contiguous().view(-1, TGT_VOCAB_SIZE), target_y.contiguous().view(-1))
        loss = get_loss(output.transpose(0, 1), target_y.transpose(0, 1))
        # -------------
        running_loss += loss.item()
    # print statistics
    final_loss = running_loss / (i + 1)
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(final_loss)
    accuracy = float(100 * correct_preds) / data_loader.get_validation_set_len()
    logger.info(f"Validation. Epoch: {epoch}, avg dev loss: {final_loss:.4f}, accuracy: {accuracy:.2f}%")
    return accuracy  # final_loss


# --------For Transformer model from SIGMORPHON 2020 Baseline----------
def train_baseline(epoch):
    """ Runs full training epoch over the training set, uses teacher forcing in training"""
    model.train()
    running_loss = 0.0
    # Get Training set in batches
    input_ids_batches, target_ids_batches, target_y_ids_batches = data_loader.get_train_set()
    # Go over each batch
    for i, (data, target, target_y) in enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches)):
        optimizer.zero_grad()
        # Get padding masks
        data = data.transpose(0, 1)
        target = target.transpose(0, 1)
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target)
        src_pad_mask, mem_pad_mask, target_pad_mask = (src_pad_mask == False).float(), (
                    mem_pad_mask == False).float(), (target_pad_mask == False).float()
        # Compute loss
        batch = (data, src_pad_mask, target, target_pad_mask)
        loss = model.get_loss(batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print statistics
    print(f"\nTrain Epoch: {epoch}, loss: {running_loss / (i + 1):.5f}")


def validation_baseline(epoch):
    """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    # Get Training set in batches
    input_ids_batches, target_ids_batches, target_y_ids_batches = data_loader.get_validation_set_tf()
    # Go over each batch
    for i, (data, target, target_y) in enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches)):
        data = data.transpose(0, 1)
        target = target.transpose(0, 1)
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target)
        src_pad_mask, mem_pad_mask, target_pad_mask = (src_pad_mask == False).float(), (
                mem_pad_mask == False).float(), (target_pad_mask == False).float()
        target_y_pad_mask = data_loader.get_padding_mask(target_y)
        # Compute loss over output (using baseline code function)
        loss = model.get_loss((data, src_pad_mask, target, target_pad_mask))
        running_loss += loss.item()
        # Compute output of model
        output = model(data, src_pad_mask, target, target_pad_mask).transpose(0, 1)
        # Get model predictions
        predictions = output.topk(1)[1].squeeze()
        target_pad_mask_test = (target_y_pad_mask == False).int()
        predictions = predictions * target_pad_mask_test
        correct_preds += torch.all(torch.eq(predictions, target_y), dim=-1).sum()
    final_loss = running_loss / (i + 1)
    accuracy = float(correct_preds) / data_loader.get_validation_set_len()
    print(f"Validation. Epoch: {epoch}, loss: {final_loss:.4f}, accuracy: {accuracy:.2f}%")
    return accuracy  # , final_loss


if __name__ == '__main__':
    eval_every = args.eval_every
    epochs_no_improve = 0
    logger.info(f"Starting training from Epoch {start_epoch + 1}")
    for epoch in range(start_epoch + 1, epochs + 1):
        # Check for early stopping
        if epochs_no_improve == args.patience:
            logger.info(
                f"Applied early stopping and stopped training. Val accuracy not improve in {args.patience} epochs")
            break
        # ---------
        train(epoch)
        # --------For Transformer model from SIGMORPHON 2020 Baseline----------
        # train_baseline(epoch)
        # ---------
        is_best = False
        curr_valid_accuracy = 0
        # Check model on validation set and get loss, every few epochs
        if epoch % eval_every == 0 and epoch > min_eval_epochs:
            epochs_no_improve += 1
            # ---------
            curr_valid_accuracy = validation(epoch)
            # --------For Transformer model from SIGMORPHON 2020 Baseline----------
            # curr_valid_accuracy = validation_baseline(epoch)
            # ---------
            # If best accuracy so far, save model as best and the accuracy
            if curr_valid_accuracy > best_valid_accuracy:
                logger.info("New best accuracy, Model saved")
                is_best = True
                best_valid_accuracy = curr_valid_accuracy
                best_valid_epoch = epoch
                epochs_no_improve = 0
        utils.save_checkpoint(model, epoch, optimizer, scheduler, curr_valid_accuracy, is_best, args.checkpoints_dir)
    utils.clean_checkpoints_dir(args.checkpoints_dir)
    logger.info(f"Finished training, best model on validation set: {best_valid_epoch},"
                f" accuracy: {best_valid_accuracy:.2f}%\n")



# if isinstance(scheduler, ReduceLROnPlateau) and get_lr() < args.min_lr:
#     logger.info(f"Applied early stopping and stopped training. current lr: {get_lr()}, min_lr {args.min_lr}")
#     break
# Train model
# seed = 0
# torch.manual_seed(seed=seed)
# if torch.cuda.is_available():
# torch.cuda.manual_seed_all(seed)
# if __name__ == '__main__':
#     eval_every = args.eval_every
#     epoch = 1
#     epochs_no_improve = 0
#     # Initialize best validation loss placeholders
#     best_valid_accuracy = -1.0
#     best_valid_epoch = 0
#     checkpoints_dir = args.checkpoints_dir
#     best_model_file = f'{checkpoints_dir}/model_best.pth'
#     for epoch in range(1, args.epochs + 1):
#         # Check for early stopping
#         if epochs_no_improve == args.patience:
#             logger.info(f"Applied early stopping and stopped training. Val accuracy not improve in {args.patience} epochs")
#             break
#         # ---------
#         train(epoch)
#         # --------For Transformer model from SIGMORPHON 2020 Baseline----------
#         # train_baseline(epoch)
#         # ---------
#         # Save model with epoch number
#         model_file = f"{checkpoints_dir}/model_{epoch}.pth"
#         torch.save(model, model_file)
#         # Check model on validation set and get loss, every few epochs
#         if epoch % eval_every == 0:
#             epochs_no_improve += 1
#             # ---------
#             curr_valid_accuracy = validation(epoch)
#             # --------For Transformer model from SIGMORPHON 2020 Baseline----------
#             # curr_valid_accuracy = validation_baseline(epoch)
#             # ---------
#             # If best accuracy so far, save model as best and the accuracy
#             if curr_valid_accuracy > best_valid_accuracy:
#                 best_valid_accuracy = curr_valid_accuracy
#                 best_valid_epoch = epoch
#                 epochs_no_improve = 0
#                 shutil.copyfile(model_file, best_model_file)
#                 logger.info("New best Loss, saved to %s" % best_model_file)
#
#     # remove unnecessary model files (disk quota limit)
#     for filename in sorted(os.listdir(checkpoints_dir)):
#         if os.path.isfile(os.path.join(checkpoints_dir, filename)) and ("best" not in filename):
#             os.remove(os.path.join(checkpoints_dir, filename))
#     logger.info(f"Finished training, best model on validation set: {best_valid_epoch}, accuracy: {best_valid_accuracy:.2f}%\n")


# model = transformer.Transformer(src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
#                                      embedding_dim=EMBEDDING_DIM,
#                                      fcn_hidden_dim=FCN_HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
#                                      dropout=DROPOUT) \
#     if (args.arch == "transformer") \
#     else \
#     pointer_generator.PointerGeneratorTransformer(src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
#                                      src_to_tgt_vocab_conversion_matrix=myTokenizer.src_to_tgt_vocab_conversion_matrix,
#                                      embedding_dim=EMBEDDING_DIM,
#                                      fcn_hidden_dim=FCN_HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
#                                      dropout=DROPOUT)
