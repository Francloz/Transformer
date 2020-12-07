import dill as pickle
import torch.optim as optim
from torch.nn.modules.transformer import *
from torchtext.data import Dataset, BucketIterator

from model import AIAYNTransformer
from model import ScheduledOptim


def truncate(src, trg, training=True) -> (torch.Tensor, torch.Tensor):
    """
    Truncates the length of the tensors so the sentences have equal length by adding padding to the right of the
    longest one. Assumes that the padding is represented by the index 1.

    :param training:
    :param src: source sentence
    :param trg: target sentence
    :return: source, target
    """
    # If training we have to pad the target tensor by one token
    if training:
        padding = torch.ones((1, trg.shape[1]), dtype=torch.long, device=trg.device)
        trg = torch.vstack((padding, trg))

    d1, d2 = src.shape[0], trg.shape[0]

    if d1 < d2:
        pad = torch.ones((d2 - d1, src.shape[1]), dtype=torch.long, device=src.device)
        src = torch.vstack((src, pad))
    if d2 < d1:
        pad = torch.ones((d1 - d2, src.shape[1]), dtype=torch.long, device=trg.device)
        trg = torch.vstack((trg, pad))

    return src, trg


def load_datasets(path, batch_size=256, verbose=False, device=torch.device('cpu')):
    """
    Loads the dataset and its configuration from a pickle file.

    :param batch_size:
    :param path: path to the pickle file which is a dictionary with the following parameters:
        'settings': settings of the dataset,
        'vocab': {'src': source vocabulary,
                  'trg': target vocabulary},
        'train': training dataset,
        'valid': validation dataset,
        # (Ignored) 'test': test dataset
    :param verbose: prints details
    :param device: device where the batches have to be moved
    :returns dictionary containing:
        'train': train dataset,
        'val': validation dataset,
        # (Ignored) 'test': test dataset
    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if verbose:
        print(f"settings: {data['settings']}\n"
              f"train: len={len(data['train'])}, example = {data['train'][0].src} -> {data['train'][0].trg}\n"
              f"valid: len={len(data['valid'])}, example = {data['valid'][0].src} -> {data['valid'][0].trg}\n"
              # f"test: len={len(data['test'])}, example = {data['test'][0].src} -> {data['test'][0].trg}\n"
              )

    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)
    # test = Dataset(examples=data['test'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator, fields


def loss_f(out, trg, eps=0.1, smoothing=True, loss=torch.nn.BCELoss(reduction='mean')) -> torch.Tensor:
    """
    Smoothed labels cross entropy loss

    :param eps: epislon of label smoothing
    :param smoothing: whether to do label smoothing or not
    :param loss: loss function to use. It must take (prediction, target), both the same shape and representing the
        probabilities of choosing each class.
    :param out: Tensor with shape LxNxV representing the probabilities of choosing each word of the vocab, where:
        L is the length of the sentence,
        N the number of batches,
        and V the size of the vocabulary
    :param trg: Tensor with shape LxN representing the index of the word in the vocabulary.
    :return: loss using cross entropy with label smoothing
    """

    if smoothing:
        K = out.shape[2]
        y_hot = F.one_hot(trg, K)
        y_ls = (1 - eps) * y_hot + eps / K
        loss = loss(out, y_ls)
    else:
        K = out.shape[2]
        y_hot = F.one_hot(trg, K)
        loss = loss(out, y_hot)

        # N = trg.shape[2]
        # out = out.view(-1)
        # trg = out.view((-1, N))
        # loss = F.cross_entropy(out, trg)
    return loss


def calc_metrics(out, trg) -> torch.Tensor:
    words = torch.Tensor([trg.shape[0] * trg.shape[1]])
    _, decisions = torch.max(out, dim=2)
    correct = torch.sum(decisions == trg).view((-1,))
    return torch.cat((words, correct, torch.zeros(2)))


def load_model(path, vocab_size, device):
    checkpoint = torch.load(path, map_location=device)
    epoch_i = checkpoint['epoch']
    settings = checkpoint['settings']

    model = AIAYNTransformer(
        vocab_size=vocab_size,
        d_model=settings['d_model'],
        n_heads=settings['n_heads'],
        n_duplicates=settings['n_duplicates'],
        d_feedforward=settings['d_feedforward'],
        p_dropout=settings['dropout']).to(device)

    model.load_state_dict(checkpoint['model'])

    optimizer_state = checkpoint['optimizer']
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizer, 2.0, settings['d_model'], 0)
    optimizer.load_state_dict(optimizer_state)

    print('[Info] Trained model state loaded.')
    return epoch_i, model, optimizer
