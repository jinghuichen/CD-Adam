#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os, pickle, math
from io import open
# import psutil 
import time

from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

import torchtext
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset


from pathlib import Path

plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 27

def serialize(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def deserialize(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def printSystemInfo():
    print("")
    print("*********************************************************************************************")
    print("Path to python interpretator:", sys.executable)
    print("Version:", sys.version)
    print("Platform name:", sys.platform)
    # print("Physical CPU processors: ", psutil.cpu_count(logical=False))
    # print("Logical CPU processors: ", psutil.cpu_count(logical=True))
    # print("Current CPU Frequncy: ", psutil.cpu_freq().current, "MHz")
    # print("Installed Physical available RAM Memory: %g %s" % (psutil.virtual_memory().total/(1024.0**3), "GBytes"))
    # print("Available physical available RAM Memory: %g %s" % (psutil.virtual_memory().available/(1024.0**3), "GBytes"))
    # print("")
    # process = psutil.Process(os.getpid())
    # mem_info = process.memory_info()

    # print("Process Resident Set (Working Set) Size: ", mem_info.rss/(1024.0 * 1024.0), "MBytes")
    # print("Virtual Memory used by process: ", mem_info.vms/(1024.0 * 1024.0), "MBytes")
    print("*********************************************************************************************")
    print("Script name: ", sys.argv[0])
    print("*********************************************************************************************")

def printTorchInfo():
    print("******************************************************************")
    print("Is CUDA avilable:", torch.cuda.is_available())
    print("GPU devices: ", torch.cuda.device_count())
    print("******************************************************************")
    print("")
    print(get_pretty_env_info())
    print("******************************************************************")

def printLayersInfo(model,model_name):
    # Statistics about used modules inside NN
    max_string_length = 0
    basic_modules = {}

    for module in model.modules():
        class_name = str(type(module)).replace("class ", "").replace("<'", "").replace("'>", "")
        if class_name.find("torch.nn") != 0:
            continue
        max_string_length = max(max_string_length, len(class_name))
 
        if class_name not in basic_modules:
            basic_modules[class_name]  = 1
        else:
            basic_modules[class_name] += 1

    print(f"Summary about layers inside {model_name}")
    print("=============================================================")
    for (layer, count) in basic_modules.items():
        print(f"{layer:{max_string_length + 1}s} occured {count:02d} times")
    print("=============================================================")
    print("Total number of parameters inside '{}' is {:,}".format(model_name, numberOfParams(model)))
    print("=============================================================")

# def numberOfParams(model):
#     total_number_of_scalar_parameters = 0
#     for p in model.parameters(): 
#         total_items_in_param = 1
#         for i in range(p.data.dim()):
#             total_items_in_param = total_items_in_param * p.data.size(i)
#         total_number_of_scalar_parameters += total_items_in_param
#     return total_number_of_scalar_parameters
def numberOfParams(model):
    return sum(p.numel() for p in model.parameters())


#=======================================================================================================
# general dataset split

def getSplitDatasets(dataset_name, batch_size, load_workers, train_workers):
    root_dir  = Path(torch.hub.get_dir()) / f'datasets/{dataset_name}'
    ds = getattr(torchvision.datasets, dataset_name)
    transform = transforms.Compose([
                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape 
                # (C x H x W) in the range [0.0, 1.0]  
                # https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.ToTensor
                transforms.ToTensor(),
                #  https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.Normalize
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_set = ds(root=root_dir, 
                   train=True, 
                   download=True, 
                   transform = transform
                  )

    test_set = ds(root=root_dir, 
                  train=False, 
                  download=True, 
                  transform = transform
                  )

    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = torch.utils.data.random_split(train_set, lengths)
    train_loaders = []
    test_loaders = []

    print(f"Total train set size for '{dataset_name}' is ", len(train_set))

    for t in range(train_workers):
        train_loader = DataLoader(
            train_sets[t],            # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        test_loader = DataLoader(
            test_set,                 # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    #==================================================================================================================================   
    classes = None

    if dataset_name == "CIFAR10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_sets, train_set, test_set, train_loaders, test_loaders, classes 


#=======================================================================================================
# Image model and data preparation

def getModel2(model_name, dataset, device):
    model_class = getattr(torchvision.models, model_name)
    model = model_class(pretrained=True).to(device)

    model.train(False)
    max_class = 0
    min_class = 0

    samples = len(dataset)
    for sample_idx in range(samples):
        input_sample, target = dataset[sample_idx]
        max_class = max(target, max_class)
        min_class = min(target, min_class)

    number_of_classes_in_dataset = max_class - min_class + 1 
    print("number_of_classes_in_dataset: ", number_of_classes_in_dataset)

    #print(f"dataset[0][0]: {dataset[0][0]}")
    out_one_hot_encoding = model(dataset[0][0].unsqueeze(0).to(device)).numel()
    #out_one_hot_encoding = model(dataset[0][0].to(device)).numel() added by Ilyas
    print("number of output class in original model: ", out_one_hot_encoding)

    final_model = torch.nn.Sequential(model,
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(out_one_hot_encoding, number_of_classes_in_dataset, bias = False)).to(device)

    # For test only
    #final_model = torch.nn.Sequential(torch.nn.Flatten(1), 
    #                                  torch.nn.Linear(32*32*3, number_of_classes_in_dataset, bias = False)).to(device)

    final_model = model

    #out_one_hot_encoding = final_model(dataset[0][0].unsqueeze(0).to(device)).numel()
    #print("number of output class in a final model: ", out_one_hot_encoding)

    return final_model

def getModel(model_name, nclass, ninput=0):
    if model_name == 'vggnet':
        from models import vgg
        model = vgg.VGG('VGG16', num_classes=nclass)
        
    elif model_name == 'resnet18':
        from models import resnet
        model = resnet.ResNet18(num_classes=nclass)
        
    elif model_name == 'wideresnet':
        from models import wideresnet
        model = wideresnet.WResNet_cifar10(num_classes=nclass, depth=16, multiplier=4)
    
    elif model_name == 'LR':
        from models import regressor
        model = regressor.LR(indim=ninput, outdim=nclass)
        
    elif model_name == 'LRText':
        from models import regressor
        model = regressor.LRText(indim=ninput, outdim=nclass)
    
    elif model_name == 'AWDLSTM':
        from models import awdrnn
        model = awdrnn.RNNModel('LSTM', ntoken=nclass, ninp=280, nhid=960, nhidlast=620, nlayers=3, 
                                dropout=0.4, dropouth=0.225, dropouti=0.4, dropoute=0.1, wdrop=0.5, tie_weights=True,
                                ldropout=0.29, n_experts=15)
                   
    elif model_name == 'Transformer':
        from models import transformer
        model = transformer.TransformerModel(ntoken=nclass, ninp=650, nhead=2, nhid=650, nlayers=2, dropout=0.5)
        
    elif model_name in ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']:
        from models import rnn
        model = rnn.RNNModel(rnn_type=model_name, ntoken=nclass, ninp=650, nhid=650, nlayers=3, dropout=0.5, tie_weights=True)
    
    elif model_name in ['TextClassifier']:
        from models import text_classifier
        model = text_classifier.TextClassificationModel(vocab_size=ninput, embed_dim=64, num_class=nclass)
    
    else:
        print('Network undefined!')

    return model


def getCriterion(nclass):
    splits = []
    if nclass > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif nclass > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
        
    from models import splitcross
    return splitcross.SplitCrossEntropyLoss(400, splits=splits, verbose=False)
        

def getSplitImageDatasets(dataset_name, batch_size, train_workers):
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root='./data/{}'.format(dataset_name), 
            train=True, 
            download=True, 
            transform = transform_train)

        test_set = torchvision.datasets.CIFAR10(
            root='./data/{}'.format(dataset_name), 
            train=False, 
            download=True, 
            transform = transform_test)
        
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    elif dataset_name == 'MNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_set = torchvision.datasets.MNIST(
            root='./data/{}'.format(dataset_name),
            train=True, 
            download=True, 
            transform=transform)
        
        test_set = torchvision.datasets.MNIST(
            root='./data/{}'.format(dataset_name),
            train=False, 
            download=True, 
            transform=transform)

        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        
    else:
        exit(f"unknown image dataset {dataset_name}")
    
    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = torch.utils.data.random_split(train_set, lengths)
    train_loaders = []
    test_loaders = []

    print(f"Total train set size for '{dataset_name}' is ", len(train_set))

    for t in range(train_workers):
        train_loader = DataLoader(
            train_sets[t],            # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=True,            # Set to True to have the data reshuffled at every epoch (default: False)
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            # collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        test_loader = DataLoader(
            test_set,                 # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            # collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_sets, train_set, test_set, train_loaders, test_loaders, classes 


#=======================================================================================================
# text classifcation data preparation
    
def getSplitTextDatasets(dataset_name, batch_size, train_workers):
    
    def yield_tokens(data_iter, tokenizer):
        for _, text in data_iter:
            yield tokenizer(text)
        
    if dataset_name == 'AG_NEWS':
        # get data snippet for vocabulary construction
        train_iter = AG_NEWS(root='./data/{}'.format(dataset_name), split='train')
        # get data split
        train_iter, test_iter = AG_NEWS(root='./data/{}'.format(dataset_name))
    
    train_set = to_map_style_dataset(train_iter)
    test_set = to_map_style_dataset(test_iter)
        
    # construct vocabulary
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    # process text to tensor
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, label_list, offsets

    # split for multi workers
    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = torch.utils.data.random_split(train_set, lengths)
    train_loaders = []
    test_loaders = []

    print(f"Total train set size for '{dataset_name}' is ", len(train_set))

    for t in range(train_workers):
        
        train_loader = DataLoader(
            train_sets[t],            # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=True,            # Set to True to have the data reshuffled at every epoch (default: False)
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=collate_batch,          
        )

        test_loader = DataLoader(
            test_set,                 # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=collate_batch, 
        )

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
    train_loader = DataLoader(
        train_set,            # dataset from which to load the data.
        batch_size=batch_size,    # How many samples per batch to load (default: 1).
        shuffle=True,            # Set to True to have the data reshuffled at every epoch (default: False)
        drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
        pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        collate_fn=collate_batch,          
    )
    
    test_loader = DataLoader(
        test_set,            # dataset from which to load the data.
        batch_size=batch_size,    # How many samples per batch to load (default: 1).
        shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
        drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
        pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        collate_fn=collate_batch,          
    )
    
    classes = list(set([label for (label, text) in train_set]))
    vocab_size = len(vocab)

    return train_loader, test_loader, train_loaders, test_loaders, classes, vocab_size

    
#=======================================================================================================
# language model and data preparation

class Vocabulary(object):
    def __init__(self, specials=['<unk>', '<pad>']):
        self.specials = specials
        self.word2idx = {}
        self.idx2word = []
        self.init()

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def build_from_iter(self, data_iter, tokenizer, with_label=False):
        self.init()
        for line in data_iter:
            for word in tokenizer(line):
                self.add_word(word)
       
    def word_to_idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['<unk>']
             
    def iter_to_idxs(self, data_iter, tokenizer):
        idss = []
        for line in data_iter:
            ids = []
            for word in tokenizer(line):
                ids.append(self.word_to_idx(word))
            idss.append(torch.tensor(ids))
        ids = torch.cat(idss)
            
        return ids
    
    def init(self):
        self.word2idx = {}
        self.idx2word = []
        for s in self.specials:
            self.add_word(s)
    
    def __len__(self):
        return len(self.idx2word)

def get_batch(source, i, bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def batchify(data, bsz):
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def getSplitMushroomDatasets(dataset_name, batch_size, train_workers):
    from dataset import MushroomDataset
    
    if dataset_name == 'mushroom':
        data = pd.read_csv('./data/mushroom/mushrooms.csv')
        
        le = LabelEncoder()
        # Convert labels to their indexes
        for col in data.columns:
            data[col] = le.fit_transform(data[col])
            
        cutoff = int(len(data)*0.8) + 1# add one to make it a round number. Easier for training.
        train_df = data.iloc[:cutoff, :]
        test_df = data.iloc[cutoff:, :]

        len_train = (len(train_df))
        len_test = (len(test_df))
        
        train_set = MushroomDataset(train_df)
        test_set = MushroomDataset(test_df)
        
        classes = {'0', '1'}
        
    else:
        exit(f'unknown dataset {dataset_name}')
        
    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = torch.utils.data.random_split(train_set, lengths)
    train_loaders = []
    test_loaders = []

    print(f"Total train set size for '{dataset_name}' is ", len(train_set))

    for t in range(train_workers):
        train_loader = DataLoader(
            train_sets[t],            # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=True,            # Set to True to have the data reshuffled at every epoch (default: False)
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        )

        test_loader = DataLoader(
            test_set,                 # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        )

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_sets, train_set, test_set, train_loaders, test_loaders, classes


def getSplitTextDatasets2(dataset_name, batch_size, train_workers):
    
    def data_process(raw_text_iter, vocab, tokenizer):
        """Converts raw text into a flat Tensor."""
        data = vocab.iter_to_idxs(raw_text_iter, tokenizer).to(torch.int64)
        return data
        
    if dataset_name == 'AG_NEWS':
        # get data snippet for vocabulary construction
        data_iter = torchtext.datasets.AG_NEWS(root='./data/{}'.format(dataset_name), split='train')
        # get data split
        train_iter, val_iter, test_iter = torchtext.datasets.AG_NEWS(root='./data/{}'.format(dataset_name))
        
    elif dataset_name == 'PennTreebank':
        data_iter = torchtext.datasets.PennTreebank(root='./data/{}'.format(dataset_name), split='train')
        train_iter, val_iter, test_iter = torchtext.datasets.PennTreebank(root='./data/{}'.format(dataset_name))
    else:
        exit(f"unknown dataset {dataset_name}")
        
    # construct vocabulary
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = Vocabulary()
    vocab.build_from_iter(data_iter, tokenizer)
    
    # create the dataset via vocab
    train_set = data_process(train_iter, vocab, tokenizer)  # [N]
    val_set = data_process(val_iter, vocab, tokenizer)
    test_set = data_process(test_iter, vocab, tokenizer)
        
    # distribute data to workers, and batchify them
    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = []
    start = 0
    for i in range(len(lengths)):
        train_sets.append(train_set)
        # end = start + lengths[i]
        # train_sets.append(train_set[start:end])
        # start = end
        
    train_loaders = []
    test_loaders = []
    
    print(f"Total train set size for '{dataset_name}' is ", len(train_set))
    
    for t in range(train_workers):
        train_loader = batchify(train_sets[i], batch_size)
        test_loader = batchify(test_set, batch_size)

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
    return train_sets, train_set, test_set, train_loaders, test_loaders, vocab 


def getSplitLanguageDatasets(dataset_name, batch_size, train_workers):
    
    def data_process(raw_text_iter, vocab, tokenizer):
        """Converts raw text into a flat Tensor."""
        data = vocab.iter_to_idxs(raw_text_iter, tokenizer).to(torch.int64)
        return data
        
    if dataset_name == 'WiKiText2':
        # get data snippet for vocabulary construction
        data_iter = torchtext.datasets.WikiText2(root='./data/{}'.format(dataset_name), split='train')
        # get data split
        train_iter, val_iter, test_iter = torchtext.datasets.WikiText2(root='./data/{}'.format(dataset_name))
        
    elif dataset_name == 'PennTreebank':
        data_iter = torchtext.datasets.PennTreebank(root='./data/{}'.format(dataset_name), split='train')
        train_iter, val_iter, test_iter = torchtext.datasets.PennTreebank(root='./data/{}'.format(dataset_name))
    else:
        exit(f"unknown dataset {dataset_name}")
        
    # construct vocabulary
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = Vocabulary()
    vocab.build_from_iter(data_iter, tokenizer)
    
    # create the dataset via vocab
    train_set = data_process(train_iter, vocab, tokenizer)  # [N]
    val_set = data_process(val_iter, vocab, tokenizer)
    test_set = data_process(test_iter, vocab, tokenizer)
        
    # distribute data to workers, and batchify them
    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = []
    start = 0
    for i in range(len(lengths)):
        train_sets.append(train_set)
        # end = start + lengths[i]
        # train_sets.append(train_set[start:end])
        # start = end
        
    train_loaders = []
    test_loaders = []
    
    print(f"Total train set size for '{dataset_name}' is ", len(train_set))
    
    for t in range(train_workers):
        train_loader = batchify(train_sets[i], batch_size)
        test_loader = batchify(test_set, batch_size)

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
    return train_sets, train_set, test_set, train_loaders, test_loaders, vocab 



#=======================================================================================================
# VISUALIZE
#=======================================================================================================
def lookAtImage(index, dataset, classes):
    image, label = dataset[index]
    plt.imshow(image.permute((1, 2, 0)), cmap='gray')
    plt.title(f'Class: {classes[label]}')
    plt.axis('off')
    plt.show()

def plotMetrics(metrics, model_name, dataset_name):
    figure, axes = plt.subplots(1, 2, figsize=(25, 5))

    # Number of epochs
    epochs = range(len(metrics['loss']))

    color = ["#e41a1c", "#377eb8", "#4daf4a", "#e41a1c", "#377eb8", "#4daf4a"]
    linestyle = ["solid", "solid", "solid", "dashed","dashed","dashed"]

    axes[0].plot(epochs, metrics['loss'], marker='o', label='train', linestyle=linestyle[0], color=color[0])
    axes[0].plot(epochs, metrics['v_loss'], marker='o', label='valid', linestyle=linestyle[1], color=color[1])
    axes[0].set_xlabel('Epochs', fontdict = {'fontsize':35})
    #axes[0].set_ylabel('Loss', fontdict = {'fontsize':35})        
    axes[0].set_title(f'Loss {model_name}@{dataset_name}')
    axes[0].legend(loc='best', fontsize=25)
    axes[0].grid()

    axes[1].plot(epochs, metrics['accuracy'], marker='o', label='train', linestyle=linestyle[2], color=color[2])
    axes[1].plot(epochs, metrics['v_accuracy'], marker='o', label='valid', linestyle=linestyle[3], color=color[3])

    axes[1].set_xlabel('Epochs', fontdict = {'fontsize':35})
    #axes[1].set_ylabel('Accuracy', fontdict = {'fontsize':35})        
    axes[1].set_title(f'Accuracy {model_name}@{dataset_name}')
    axes[1].legend(loc='best', fontsize=25)
    axes[1].grid()

    plt.show(figure)
    best = max(metrics['v_accuracy']) * 100
    print(f'Best validation accuracy {best:.2f}%')
    
    figure.tight_layout()
    save_to = f"plots-{model_name}-{dataset_name}.pdf"
    figure.savefig(save_to, bbox_inches='tight')
    print("Image is saved into: ", save_to)

#=======================================================================================================
# OPERATE ON NN PARAMETERS
#=======================================================================================================
def setupToZeroAllParams(model):
    for p in model.parameters(): 
        p.data.zero_()

def setupAllParamsRandomly(model):
    #setupToZeroAllParams(model)

    seed = 12
    torch.manual_seed(seed)

    # torch.nn.init.xavier_normal_(model.parameters)
    # for p in model.parameters():
    #     sz = p.data.flatten(0).size()
    #     torch.nn.init.xavier_normal_(p.weight)
    #     p.data.flatten(0)[:] = 2 * (torch.rand(size = sz) - 0.5)

def setupAllParams(model, params):
    i = 0
    for p in model.parameters():
        p.data.flatten(0)[:] = params[i].flatten(0)
        i += 1

def getAllParams(model):
    params = []
    for p in model.parameters(): 
        params.append(p.data.detach().clone())
    return params

def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z


def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z


def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z
