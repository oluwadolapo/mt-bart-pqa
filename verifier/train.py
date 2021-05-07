import time
import math
import torch

from utils import flat_accuracy
from utils import format_time

def train_lstm_cnn(args, model, iterator, optimizer, criterion):
    print("")
    print('Training...')
    
    t0 = time.time()
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    # import IPython; IPython.embed(); exit(1)
    for step, batch in enumerate(iterator):
        optimizer.zero_grad()
        # Progress update every 1000 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress
            print(' Batch {:>5,} of {:>5,}.   Elapsed: {:}.'.format(step, len(iterator), elapsed))

        if args.model_type == "lstm":
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
        elif args.model_type == "cnn":
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)

        # Move predictions and labels to CPU
        predictions = predictions.to(dtype=int).detach().cpu().numpy()
        labels = batch.label.to(dtype=int).detach().cpu().numpy()

        acc = flat_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    # Measure how long the validation run took.
    training_time = format_time(time.time() - t0)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), training_time

def validate_lstm_cnn(args, model, iterator, criterion):
    print("")
    print('Running Validation...')

    t0 = time.time()

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            if args.model_type == "lstm":
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
                loss = criterion(predictions, batch.label)
            elif args.model_type == "cnn":
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
            
            # Move predictions and labels to CPU
            predictions = predictions.to(dtype=int).detach().cpu().numpy()
            labels = batch.label.to(dtype=int).detach().cpu().numpy()

            acc = flat_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), validation_time

"""
def evaluate(model, iterator, criterion):
    print("")
    print('Running Evaluation...')

    t0 = time.time()

    epoch_loss = 0
    epoch_acc = 0

    # Tracking variables
    total_test_accuracy = 0
    total_test_precision = 0
    total_test_recall = 0
    total_test_f1 = 0

    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
            # Move predictionss and labels to CPU
            predictions = predictions.detach().cpu().numpy()
            labels = batch.label.to('cpu').numpy()
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), validation_time

"""

def train_bart(args, model, class_h, train_dataloader, device, optimizer, scheduler):
    print("")
    print('Training...')
    
    t0 = time.time()

    model.train()
    # Reset the total loss for this epoch.
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        # Progress update every 100 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress
            print(' Batch {:>5,} of {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        model.zero_grad()
        logits = model(input_ids = batch[0].to(device),
                            labels = batch[4].to(device),
                            attention_mask = batch[1].to(device),
                            decoder_input_ids = batch[2].to(device),
                            decoder_attention_mask = batch[3].to(device),
                            is_training = True,
                            device = device)
        _, loss = class_h(logits, batch[4].to(device))
        total_train_loss += loss.mean().detach().cpu()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Update parameters
        optimizer.step()

        # Update the learning rate
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long the validation run took.
    training_time = format_time(time.time() - t0)

    return avg_train_loss, training_time


def validate_bart(model, class_h, eval_dataloader, device):
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    # Tracking variables
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in eval_dataloader:

        with torch.no_grad():
            logits = model(input_ids = batch[0].to(device),
                             attention_mask = batch[1].to(device),
                             decoder_input_ids = batch[2].to(device),
                             decoder_attention_mask = batch[3].to(device),
                             labels = batch[4].to(device),
                             is_training = True,
                             device = device)
 
            _, val_loss = class_h(logits, batch[4].to(device))
        # Accumulate the validation loss.
        #print('val_loss', val_loss)
        total_eval_loss += val_loss.mean().detach().cpu()

    # Calculate the average loss over all the batches.
    avg_val_loss = total_eval_loss / len(eval_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    return avg_val_loss, validation_time