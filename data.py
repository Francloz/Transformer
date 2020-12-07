import dill as pickle
from torchtext.data import Dataset, BucketIterator


def load_datasets(path, verbose=False):
    """
    Loads the dataset and its configuration from a pickle file.

    :param path: path to the pickle file which is a dictionary with the following parameters:
        'settings': settings of the dataset,
        'vocab': {'src': source vocabulary,
                  'trg': target vocabulary},
        'train': training dataset,
        'valid': validation dataset,
        # (Ignored) 'test': test dataset
    :param verbose: prints details
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

    return {
        'train': train,
        'val': val,
        # 'test': test
    }


def get_dataloaders(data, batch_size, device):
    """

    :param data: data size
    :param batch_size: batch size
    :param device: device where the batches have to be moved
    :return:
    """

    train_iterator = BucketIterator(data['train'], batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(data['val'], batch_size=batch_size, device=device)
    return train_iterator, val_iterator

# if __name__ == "__main__":
#     train, val = get_dataloaders(load_datasets("dataset/m30k_deen_shr.pkl", verbose=True), 32, 'cpu')
#     for b in train:
#         src, trg = b.src, b.trg
#         print(src.shape)
#         pass
#     load_datasets("dataset/preprocessed.pkl", verbose=True)
#
#     pass
