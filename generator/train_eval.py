import math
import time
import torch

from utils import format_time

def train(args, model, train_dataloader, device, optimizer, scheduler):
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
        _, loss = model(input_ids = batch[0].to(device),
                             attention_mask = batch[1].to(device),
                             decoder_input_ids = batch[2].to(device),
                             decoder_attention_mask = batch[3].to(device),
                             is_training = True)
        #print('training', loss)
  
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
    avg_train_ppl = math.exp(avg_train_loss)

    # Measure how long the validation run took.
    training_time = format_time(time.time() - t0)

    return avg_train_loss, avg_train_ppl, training_time


def eval(model, eval_dataloader, device):
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
            _, val_loss = model(input_ids = batch[0].to(device),
                             attention_mask = batch[1].to(device),
                             decoder_input_ids = batch[2].to(device),
                             decoder_attention_mask = batch[3].to(device),
                             is_training = True)
 
            
        # Accumulate the validation loss.
        #print('val_loss', val_loss)
        total_eval_loss += val_loss.mean().detach().cpu()

    # Calculate the average loss over all the batches.
    avg_val_loss = total_eval_loss / len(eval_dataloader)
    avg_val_ppl = math.exp(avg_val_loss)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    return avg_val_loss, avg_val_ppl, validation_time