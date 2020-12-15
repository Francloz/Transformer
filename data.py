import dill as pickle
import numpy as np
from torchtext.data import Dataset as TextDataset


def load_datasets_pkl(path, verbose=False):
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

    train = TextDataset(examples=data['train'], fields=fields)
    val = TextDataset(examples=data['valid'], fields=fields)
    # test = Dataset(examples=data['test'], fields=fields)

    return {
        'train': train,
        'val': val,
        # 'test': test
    }


class TranslationDataset:
    """
    Dataset for text translation using the following format:

    Dataset size: Nx(2L + 1) where N is the number of samples and L is the maximum length of a sentence.

    A row has 2L + 1 elements:
        row[0] = maximum size between the size of the source sentence and the target sentence.
        row[1:L+1] = encoded tokens of the source sentence.
        row[L+2:] = encoded tokens of the target sentence.

    If a sentence is smaller than L tokens in length, the rest is filled with the code of the padding token, usually 1.
    This way we can force source and target to have same length during training.

    Example:
    A dataset of L=5 with to have the following shape:

    [[4, 24, 451, 23, 1, 1, 32556, 23, 51, 53, 1],
    [ 2,  5,   3,  1, 1, 1,   356,  1,  1,  1, 1]]

    Here the source sentence would be [ 24, 451, 23] and the target sentence would be [32556, 23, 51, 53] and 4 is
    the length of the target sentence. If we get the sample at index 0 from this dataset, we would get:

    src: [   24, 451, 23,  1]
    trg: [32556,  23, 51, 53]

    NOTE: IF USED TOGETHER WITH THE BUCKET ITERATOR, THE SAMPLES MUST BE ORDERED ACCORDING TO THE FIRST COLUMN IN
    ASCENDING ORDER.
    """

    def __init__(self, data):
        """
        :param data: the shape of the data must be  Nx(2L + 1) where N is the number of samples and L is the maximum
        length of a sentence
        """
        self.data = data
        self.length = self.data.shape[0]
        self.middle = (self.data.shape[1] - 1) // 2

    def __len__(self):
        """
        Returns the number of samples of the dataset
        :return: number of rows in the dataset
        """
        return self.length

    def __getitem__(self, idx):
        """
        Returns the samples given in the idx using the minimum number of padding in order to return all samples with the same length.
        The source and the target are too the same length.

        :param idx: indices to extract from the dataset
        :return: source samples, target samples
        """
        elements = self.data[idx]
        length = np.max(elements[:, 0])
        src = elements[:, 1:length + 1]
        trg = elements[:, 1 + self.middle:length + self.middle + 1]
        return src, trg


class BucketIterator:
    """
    Bucket iterator. This bucket iterator attempts to return every batch containing the same number of tokens.

    For this, we need the dataset to be sorted according to sample length. This way, we can group the samples into
    buckets with equal length. This is transmitted to the bucket iterator with array of shape Bx2 where B is the number
    of buckets and there are two columns, the index at which the new length starts and the said length.

    The way the iterator works is by using a random uniform distribution among all samples in the dataset, but once
    picked the bucket of the first sample. We prioritize picking samples of the same bucket. If the bucket runs out of
    samples, we pick samples from the previous bucket, and if that runs out too, we pick samples of the next bucket.

    We do this till the number of samples multiplied by the maximum length of a sample picked is approximately the
    desired number of tokens.
    """

    def __init__(self, counts, total, ntokens=1, drop_last=True):
        """
        :param counts: file containing the distribution of the dataset
        :param total: total size of the file
        :param ntokens: number of tokens to get
        :param drop_last: whether to drop the last incomplete batch
        """
        self._indexes = counts[:, 0]
        self._lengths = counts[:, 1]

        self.block_sizes = np.append(self._indexes[1:], total) - self._indexes

        self._all_indexes = []
        for (start, amount) in zip(self._indexes, self.block_sizes):
            arr = np.arange(start=start, stop=start + amount)
            np.random.shuffle(arr)
            self._all_indexes.append(arr)

        self.ntokens = ntokens
        self.drop_last = drop_last
        self._p = self.block_sizes / total

    def __len__(self) -> int:
        """
        :return: returns the total amount of samples
        """
        return sum(self.block_sizes)

    def __iter__(self):
        """Resets/Initializes this class to function as a correct iterator."""
        self._all_indexes = []
        for (start, amount) in zip(self._indexes, self.block_sizes):
            arr = np.arange(start=start, stop=start + amount)
            np.random.shuffle(arr)
            self._all_indexes.append(arr)
        self._p = self.block_sizes / sum(self.block_sizes)
        return self

    def __next__(self):
        """
        Fetches the indices that correspond for the next batch.
        :return: indices of the next batch

        NOTE: It does not keep track of the amount of samples on each bucket, because of this the code calls len on
        the buckets very often to be able to calculate the probability of picking a sample from a bucket if each sample
        has uniform probability. This might be changed in future versions for clarity and optimization.
        """

        # Check whether we have the necessary number of samples to continue
        accum = 0
        for arr in self._all_indexes:
            accum += len(arr) * 100
            if accum > self.ntokens:
                break
        if accum == 0 or (accum < self.ntokens and self.drop_last):
            raise StopIteration

        # We pick from what bucket we will be picking the next batch
        block_idx = np.random.choice(a=len(self._indexes), p=self._p)
        chosen_blocks = [block_idx, block_idx]
        block_length = self._lengths[block_idx]
        approximation = self.ntokens // block_length
        len_possibilities = len(self._all_indexes[block_idx])

        # If the bucket is empty, we start adding whichever buckets are closest to start filling in the batch
        while len_possibilities < approximation and chosen_blocks[1] - chosen_blocks[0] < len(self._indexes):
            if block_idx >= (chosen_blocks[1] - chosen_blocks[0]) // 2 and chosen_blocks[1] + 1 < len(self._indexes):
                chosen_blocks[1] += 1
                len_possibilities += len(self._all_indexes[chosen_blocks[1]])
                block_length = self._lengths[chosen_blocks[1]]
            else:
                chosen_blocks[0] -= 1
                len_possibilities += len(self._all_indexes[chosen_blocks[0]])

            approximation = self.ntokens // block_length

        samples = []
        total = sum([len(self._all_indexes[idx]) for idx in range(chosen_blocks[0], chosen_blocks[1] + 1)])
        num_chosen = chosen_blocks[1] - chosen_blocks[0] + 1

        # Once choosen which buckets are necessary, we start grabbing samples from them
        if num_chosen > 1:
            # If we pick the samples from more than one bucket, we have to pick them one by one to make sure that the
            # probabilities are uniform. This makes the batch require longer to fetch. This is no problem however
            # because this does not happen often in real dataset situations. Either way, real uniform probability might
            # be discarded in the future in exchange of faster computation.
            while len(samples) < approximation:
                amounts = [len(self._all_indexes[idx]) for idx in range(chosen_blocks[0], chosen_blocks[1] + 1)]
                if sum(amounts) == 0:
                    return np.array(samples)
                probs = [p / total for p in amounts]
                choice = np.random.choice(num_chosen, p=probs)
                samples.append(self._all_indexes[chosen_blocks[0] + choice][0])
                self._all_indexes[chosen_blocks[0] + choice] = self._all_indexes[chosen_blocks[0] + choice][1:]
                total -= 1
            batch = np.array(samples)
        else:
            batch = self._all_indexes[chosen_blocks[0]][:approximation]
            self._all_indexes[chosen_blocks[0]] = self._all_indexes[chosen_blocks[0]][approximation:]
            total -= len(batch)

        self._p = np.array([len(arr) / total for arr in self._all_indexes])
        return batch


class TextDataLoader:
    """
    Dataloader that uses both the TranslationDataset to store and fetch samples and the BucketIterator to iterate over
    the dataset batch by batch.
    """

    def __init__(self, data, counts, ntokens_per_batch, drop_last):
        """
        :param data: data for the TextDataset, it has to be sorted by the first column and match the counts data
        :param counts: counts for the BucketIterator
        :param ntokens_per_batch: number of tokens to fetch with each batch
        :param drop_last: whether to drop the last incomplete batch or not
        """
        self.dataset = TranslationDataset(data)
        self.batch_sampler = TextDataLoader(counts, data.shape[0], ntokens_per_batch, drop_last)

    def __iter__(self):
        """
        Initializes the batch sampler so that the iteration is correct.
        :return: self
        """
        iter(self.batch_sampler)
        return self

    def __next__(self):
        """
        Grabs the next batch from the dataset.
        :return: next batch
        """
        idx = next(self.batch_sampler)
        return self.dataset[idx]


def load_datasets_npy(path, ntokens=25000, mmap_mode=None):
    """
    Loads the dataset and its configuration from a pickle file.

    :param mmap_mode: mmap_mode of the numpy array containing the dataset. For more information check the numpy.load
        operation
    :param ntokens: number of tokens (from the source language) to load as a batch. From this parameter the dataloader
        decides how many samples to get. Longer sentences mean fewer samples in a batch.
    :param path: path to the base file
        ex.:  Real file /path/to/file/fren-train.src.trg has base file /path/to/file/fren-

    :returns dictionary containing:
        'train': train dataset,
        'val': validation dataset
    """

    train_data = np.load(path + "train" + ".npy", mmap_mode=mmap_mode)
    train_count = np.load(path + "train" + ".count.npy")

    val_data = np.load(path + "val" + ".npy", mmap_mode=mmap_mode)
    val_count = np.load(path + "val" + ".count.npy")

    train = TextDataLoader(train_data, train_count, ntokens, True)
    val = TextDataLoader(val_data, val_count, ntokens, True)

    return {
        'train': train,
        'val': val
    }


if __name__ == "__main__":
    load_datasets_npy(path="G:\\Documents\\Datasets\\Tranformer\\tmp\\fren\\fren-")
