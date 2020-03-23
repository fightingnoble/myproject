import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler

NUM_TRAIN = 58000
NUM_VAL = 2000


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def getmnist(args, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../datasets', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=args.train_batch_size, shuffle=False, sampler=ChunkSampler(NUM_TRAIN, 0), **kwargs)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../datasets', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=args.test_batch_size, shuffle=False, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN), **kwargs)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../datasets', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader

# 2020.0223: add cross validation, remove cross validation
#     train_db, val_db = torch.utils.data.random_split(
#         torchvision.datasets.MNIST(args.dataset_root, train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ])), [NUM_TRAIN, NUM_VAL])
#
#     train_loader = torch.utils.data.DataLoader(
#         train_db,
#         batch_size=args.train_batch_size, shuffle=False, **kwargs)
#
#     val_loader = torch.utils.data.DataLoader(
#         val_db,
#         batch_size=args.test_batch_size, shuffle=False, **kwargs)
