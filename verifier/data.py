from torchtext import data, datasets
import torchtext.vocab as voc
from torchtext.vocab import GloVe, FastText, CharNGram
import torch

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import random
import numpy as np
import sys


def train_lstm_cnn_data(args, device):
    if args.local_run:
        if args.model_type == "lstm":
            TEXT = data.Field(tokenize = 'spacy', include_lengths=True, fix_length=20)
        elif args.model_type == "cnn":
            TEXT = data.Field(tokenize = 'spacy', batch_first = True, fix_length=20)

    else:
        if args.model_type == "lstm":
            TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
        elif args.model_type == "cnn":
            TEXT = data.Field(tokenize = 'spacy', batch_first = True)
    
    LABEL = data.LabelField(dtype = torch.float)

    fields = {'quest_rev': ('text', TEXT), 'is_answerable': ('label', LABEL)}
    train_data, valid_data = data.TabularDataset.splits(path = args.data_path,
                                    train = args.train_file_name,
                                    validation = args.val_file_name,
                                    format = 'json',
                                    fields = fields)

    #train_data, valid_data = train_data.split(random_state = random.seed(args.random_seed))

    if args.from_scratch:
        TEXT.build_vocab(train_data, max_size = args.max_voc_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
        LABEL.build_vocab(train_data)
    else:
        custom_embeddings = voc.Vectors(name = args.load_emb_path,
                                  #cache = 'custom_embeddings',
                                  unk_init = torch.Tensor.normal_)
        TEXT.build_vocab(train_data, 
                 max_size = args.max_voc_size, 
                 vectors = custom_embeddings)

        LABEL.build_vocab(train_data)

    # Save vocabulary
    with open(args.save_voc_path, 'w+') as f:
        for token, index in TEXT.vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = args.train_batch_size, 
        sort_within_batch=True,
        sort_key = lambda x: x.text,
        device = device)
    
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    return train_iterator, valid_iterator, TEXT.vocab, UNK_IDX, PAD_IDX

    

def test_lstm_cnn_data(args, device):
    if args.local_run:
        if args.model_type == "lstm":
            TEXT = data.Field(tokenize = 'spacy', include_lengths=True, fix_length=20)
        elif args.model_type == "cnn":
            TEXT = data.Field(tokenize = 'spacy', batch_first = True, fix_length=20)

    else:
        if args.model_type == "lstm":
            TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
        elif args.model_type == "cnn":
            TEXT = data.Field(tokenize = 'spacy', batch_first = True)
    
    LABEL = data.LabelField(dtype = torch.float)

    fields = {'quest_rev': ('text', TEXT), 'is_answerable': ('label', LABEL)}
    train_data, test_data = data.TabularDataset.splits(path=args.data_path,
                                                        train=args.train_file_name,
                                                        test=args.test_file_name,
                                                        format='json',
                                                        fields=fields)
    
    """
    custom_embeddings = voc.Vectors(name = args.load_emb_path,
                                  #cache = 'custom_embeddings',
                                  unk_init = torch.Tensor.normal_)
    
    
    TEXT.build_vocab(train_data,
                max_size = args.max_voc_size, 
                vectors = custom_embeddings)
    """
    TEXT.build_vocab(train_data, max_size = args.max_voc_size, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)


    test_iterator = data.BucketIterator(test_data, 
                            train = False,
                            batch_size = args.predict_batch_size, 
                            sort_within_batch=True,
                            sort_key = lambda x: x.text,
                            device = device)
    return test_iterator, TEXT.vocab


def bart_tokenize_batch(quest_rev, answers, tokenizer, max_len):
    # import IPython; IPython.embed(); exit(1)
    encoder_input = tokenizer.batch_encode_plus(
    quest_rev,
    pad_to_max_length = True,
    max_length = max_len,
    truncation = True)

    decoder_input = tokenizer.batch_encode_plus(
    answers,
    pad_to_max_length = True,
    max_length = max_len,
    truncation = True)

    return encoder_input, decoder_input

def bart_prepare_data(df):
    questions = [q if q.endswith("?") else q+"?" for q in df.questionText]
    reviews = [r for r in df.review_snippets]

    questions = [q.lower() + " </s>" for q in questions]
    reviews = [[p.lower() for p in r] for r in reviews]

    quest_rev = []
    for i, q in enumerate(questions):
        for r in reviews[i]:
            q = q + " " + r
        q += " </s>"
        quest_rev.append(q)

    return quest_rev


def bart_training_data(args, tokenizer):
    df = pd.read_json(args.data_path, orient='split')
    #df = df.head(49500)

    if args.local_run:
        df = df.head(12)
        split = 9
    else:
        if args.data_size == 50000:
            df = df.head(49000)
            split = 45001
        elif args.data_size == 30000:
            df = df.head(30000)
            split = 25001

    quest_rev = bart_prepare_data(df)

    # Get the labels from the DataFrame, and convert from booleans to ints.
    labels = df.is_answerable.to_numpy().astype(float)

    encoder_input, decoder_input = bart_tokenize_batch(quest_rev[:split], quest_rev[:split], tokenizer, args.max_len)
    train_inputs, train_input_mask = encoder_input["input_ids"], encoder_input["attention_mask"]
    train_targets, train_target_mask = decoder_input["input_ids"], decoder_input["attention_mask"]
    train_label = labels[:split]

    encoder_input, decoder_input = bart_tokenize_batch(quest_rev[split:], quest_rev[split:], tokenizer, args.max_len)
    validation_inputs, validation_input_mask = encoder_input["input_ids"], encoder_input["attention_mask"]
    validation_targets, validation_target_mask = decoder_input["input_ids"], decoder_input["attention_mask"]
    validation_label = labels[split:]

    # Convert all inputs and targets into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_targets = torch.tensor(train_targets)
    validation_targets = torch.tensor(validation_targets)

    train_input_mask = torch.tensor(train_input_mask)
    validation_input_mask = torch.tensor(validation_input_mask)
    train_target_mask = torch.tensor(train_target_mask)
    validation_target_mask = torch.tensor(validation_target_mask)

    train_label = torch.tensor(train_label)
    validation_label = torch.tensor(validation_label)

    batch_size = args.train_batch_size

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_input_mask, train_targets, train_target_mask, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    validation_data = TensorDataset(validation_inputs, validation_input_mask, validation_targets, validation_target_mask, validation_label)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def bart_testing_data(args, tokenizer):
    df = pd.read_json(args.data_path, orient='split')
    #df = df.head(49500)

    if args.local_run:
        df = df.head(50)

    quest_rev = bart_prepare_data(df)

    # Get the labels from the DataFrame, and convert from booleans to ints.
    labels = df.is_answerable.to_numpy().astype(float)

    encoder_input, decoder_input = bart_tokenize_batch(quest_rev, quest_rev, tokenizer, args.max_len)
    test_inputs, test_input_mask = encoder_input["input_ids"], encoder_input["attention_mask"]
    test_targets, test_target_mask = decoder_input["input_ids"], decoder_input["attention_mask"]
    test_label = labels


    # Convert all inputs and targets into torch tensors, the required datatype
    # for our model.
    test_inputs = torch.tensor(test_inputs)
    test_targets = torch.tensor(test_targets)
    test_input_mask = torch.tensor(test_input_mask)
    test_target_mask = torch.tensor(test_target_mask)
    test_label = torch.tensor(test_label)

    batch_size = args.train_batch_size

    # Create the DataLoader for our testing set
    test_data = TensorDataset(test_inputs, test_input_mask, test_targets, test_target_mask, test_label)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader
