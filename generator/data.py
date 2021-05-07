import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def tokenize_batch(quest_rev, answers, tokenizer, max_len):
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

def prepare_data(df):
    questions = [q if q.endswith("?") else q+"?" for q in df.question]
    reviews = [r for r in df.passages]
    answers = [a for a in df.answers]

    questions = [q.lower() + " </s>" for q in questions]
    reviews = [[p.lower() for p in r] for r in reviews]
    answers = ["<s> " + a.lower() + " </s>" for a in answers]

    quest_rev = []
    for i, q in enumerate(questions):
        for r in reviews[i]:
            q = q + " " + r
        quest_rev.append(q)

    return quest_rev, answers


def training_data(args, tokenizer):
    df = pd.read_json(args.data_path, orient='split')
    #df = df.head(49500)

    if args.local_run:
        df = df.head(12)
        split = 9
    else:
        if args.data_size == 50000:
            df = df.head(49500)
            split = 45001
        elif args.data_size == 30000:
            df = df.head(30000)
            split = 25001

    quest_rev, answers = prepare_data(df)

    encoder_input, decoder_input = tokenize_batch(quest_rev[:split], answers[:split], tokenizer, args.max_len)
    train_inputs, train_input_mask = encoder_input["input_ids"], encoder_input["attention_mask"]
    train_targets, train_target_mask = decoder_input["input_ids"], decoder_input["attention_mask"]

    encoder_input, decoder_input = tokenize_batch(quest_rev[split:], answers[split:], tokenizer, args.max_len)
    validation_inputs, validation_input_mask = encoder_input["input_ids"], encoder_input["attention_mask"]
    validation_targets, validation_target_mask = decoder_input["input_ids"], decoder_input["attention_mask"]

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

    batch_size = args.train_batch_size

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_input_mask, train_targets, train_target_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    validation_data = TensorDataset(validation_inputs, validation_input_mask, validation_targets, validation_target_mask)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader
