import argparse
import os
import time

from tqdm import tqdm

from model import *
from utils import *


def eval_epoch(model, validation_data):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    total_metrics = torch.zeros(4)
    k = 0

    with torch.no_grad():
        for batch in validation_data:
            k += 0
            # prepare data
            src, trg = batch.src, batch.trg
            src, trg = truncate(src, trg)

            # forward
            out = model(src, trg)
            loss = loss_f(out, trg)
            total_loss += loss.item()
            metrics = calc_metrics(out, trg)
            total_metrics += metrics

    if metrics[0] > 0:
        loss_per_word = total_loss / k  # / total_metrics[0]
        accuracy = total_metrics[1] / total_metrics[0]
    else:
        loss_per_word = -1
        accuracy = -1
    return loss_per_word, accuracy


def train_epoch(model, training_data, optimizer, print_freq=10, nbatches=-1):
    """
    Trains one epoch.

    :param nbatches: Maximum number of batches to iterate over. A negative number will be automatically changed to the
        length of the dataloader.
    :param print_freq: Frequency of prints to check how the loss and accuracy are going.
    :param model: transformer
    :param training_data: training dataloader (torchtext.data Batch iterator expected)
    :param optimizer: optimizer
    :return: loss, accuracy
    """
    nbatches = nbatches if nbatches >= 0 else len(train_data)
    model.train()
    total_loss, metrics = 0, torch.zeros(4)
    partial_loss, partial_metrics = 0, torch.zeros(4)
    total_metrics = torch.zeros(4)

    k = 0
    for batch in tqdm(training_data, mininterval=1, leave=False):
        # for batch in training_data:
        k += 1
        if k > nbatches:
            break

        # prepare data
        src, trg = batch.src, batch.trg
        src, trg = truncate(src, trg)

        # forward
        optimizer.zero_grad()
        out = model(src, trg)
        loss = loss_f(out, trg)
        loss.backward()
        optimizer.step_and_update_lr()

        loss = loss.item()
        total_loss += loss

        metrics = calc_metrics(out, trg)
        total_metrics += metrics

        # if print_freq > 0:
        #     partial_loss += loss
        #     partial_metrics += metrics

        # if k % print_freq == 0:
        #     loss_per_word = total_loss / print_freq  # / total_metrics[0]
        #     accuracy = partial_metrics[1] / partial_metrics[0]
        #     print(f"\r{k}, {accuracy}, {loss_per_word}", end="\n")
        #     partial_loss, partial_metrics = 0, torch.zeros(4)

    if metrics[0] > 0:
        loss_per_word = total_loss / k  # / total_metrics[0]
        accuracy = total_metrics[1] / total_metrics[0]
    else:
        loss_per_word = -1
        accuracy = -1
    return loss_per_word, accuracy


def train(training_data, validation_data, device, settings,
          vocab_size, epochs=1, load_file="", save_file="model", n_warmup_steps=128000):
    """
    Trains the model for.

    :param training_data: training dataloader
    :param validation_data: validation dataloader
    :param device: torch device
    :param epochs: number of epochs
    :param load_file: path to the file to load
    :param save_file: saves the model to this file
    :return:
    """

    def print_performances(header, loss, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=math.exp(min(loss, 100)),
            accu=100 * accu, elapse=(time.time() - start_time) / 60))

    if os.path.isfile(load_file):
        current, model, optimizer = load_model(load_file, device, vocab_size)
    else:
        # import psutil
        # process = psutil.Process(os.getpid())
        # m1 = process.memory_info().rss

        current = 0
        model = AIAYNTransformer(
            vocab_size=vocab_size,
            d_model=settings['d_model'],
            n_heads=settings['n_heads'],
            n_duplicates=settings['n_duplicates'],
            d_feedforward=settings['d_feedforward'],
            p_dropout=settings['dropout']).to(device)

        # m2 = process.memory_info().rss
        # print((m2 - m1) / 2 ** 30)
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        optimizer = ScheduledOptim(optimizer, 2.0, settings['d_model'], n_warmup_steps)

    valid_losses = []
    for epoch_i in range(current, epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer)
        print_performances('Training', train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data)
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'model': model.state_dict(), 'settings': settings,
                      'optimizer': optimizer.state_dict()}
        model_name = save_file + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
        torch.save(checkpoint, model_name)

        model_name = save_file + '.chkpt'
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, model_name)
            print('    - [Info] The checkpoint file has been updated.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', type=str, default="dataset/m30k_deen_shr.pkl",
                        help="Path to the all in one pickled data")
    parser.add_argument('-model', default='base', choices=['base', 'big'])
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=128000)
    parser.add_argument('-log', type=str, default="")
    parser.add_argument('-save_model', type=str, default="")
    parser.add_argument('-cuda', default=False, action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    # device = torch.device('cuda')
    settings = BASE_MODEL if args.model == 'base' else BIG_MODEL
    n_warmup_steps = args.n_warmup_steps
    batch_size = args.batch_size

    train_data, val_data, fields = load_datasets(args.data_pkl, batch_size=batch_size, device=device)
    vocab_size = len(fields['src'].vocab)

    train(train_data, val_data, device, n_warmup_steps=n_warmup_steps,
          settings=settings, epochs=400, load_file="saved_models/model.chkpt",
          save_file='saved_models/test', vocab_size=vocab_size)
