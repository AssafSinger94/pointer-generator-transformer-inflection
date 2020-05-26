import logging
import os
import shutil
import sys
import time
from datetime import timedelta

import torch
from torch.optim.lr_scheduler import LambdaLR

import pointer_generator
import transformer


class WarmupInverseSquareRootSchedule(LambdaLR):
    """ Linear warmup and then inverse square root decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_factor = warmup_steps**0.5
        super(WarmupInverseSquareRootSchedule,
              self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5


class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (record.levelname, time.strftime('%x %X'),
                                   timedelta(seconds=elapsed_seconds))
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def get_logger():
    '''
    create logger and output to file and stdout
    '''
    log_formatter = LogFormatter()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(log_formatter)
    logger.addHandler(stream)
    return logger

def maybe_mkdir(filename):
    '''
    maybe mkdir
    '''
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def save_checkpoint(model, epoch, optimizer, scheduler, val_accuracy, is_best, checkpoints_dir):
    """
    Save checkpoint of model at current epoch, if new best model, saves checkpoint as best.
    Saves state of  model, epoch, optimizer and scheduler.
    """
    # Create the checkpoint folder in not exists
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    checkpoint = {
        'epoch': epoch,
        'val_accuracy': val_accuracy,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    model_path = f"{checkpoints_dir}/model_{epoch}.pth"
    best_model_path = f'{checkpoints_dir}/model_best.pth'
    torch.save(checkpoint, model_path)
    if is_best:
        shutil.copyfile(model_path, best_model_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, logger):
    """
    Load checkpoint of model from checkpoint path. Used for training.
    Loads state of model, epoch, optimizer and scheduler.
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f" Trying to resume training bot file {checkpoint_path} not exists,\n"
                    f" starting training from scratch")
        return model, optimizer, scheduler, 0, -1.0
    logger.info(f"resume training, loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        logger.debug("Model hyperparameters do not match loaded checkpoint")
    optimizer.load_state_dict(checkpoint['optimizer'])
    try:
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
        logger.debug("Scheduler does not match loaded checkpoint")
    start_epoch = checkpoint['epoch']
    val_accuracy = checkpoint['val_accuracy']
    return model, optimizer, scheduler, start_epoch, val_accuracy

def load_model(model, checkpoint_path, logger):
    """
    Load checkpoint of model from checkpoint path. Used for generating prediction files.
    Only Loads state of model.
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f" Trying to reload checkpoint from pretraining but file {checkpoint_path} not exists,\n"
                    f" starting training from scratch")
    else:
        checkpoint = torch.load(checkpoint_path)
        state_dict = model.state_dict()
        state_dict.update(checkpoint['state_dict'])
        model.load_state_dict(state_dict)
    return model

def build_model(arch, src_vocab_size, tgt_vocab_size, embedding_dim, fcn_hidden_dim,
                num_heads, num_layers, dropout, src_to_tgt_vocab_conversion_matrix):
    """
    Builds model.
    """
    model = transformer.Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                    embedding_dim=embedding_dim,
                                    fcn_hidden_dim=fcn_hidden_dim, num_heads=num_heads, num_layers=num_layers,
                                    dropout=dropout) \
        if (arch == "transformer") \
        else \
        pointer_generator.PointerGeneratorTransformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                                      src_to_tgt_vocab_conversion_matrix=src_to_tgt_vocab_conversion_matrix,
                                                      embedding_dim=embedding_dim,
                                                      fcn_hidden_dim=fcn_hidden_dim, num_heads=num_heads,
                                                      num_layers=num_layers,
                                                      dropout=dropout)
    return model


def clean_checkpoints_dir(checkpoints_dir):
    """
    Remove unnecessary model checkpoints (disk quota limit)
    """
    for filename in sorted(os.listdir(checkpoints_dir)):
        if os.path.isfile(os.path.join(checkpoints_dir, filename)) and ("best" not in filename):
            os.remove(os.path.join(checkpoints_dir, filename))





