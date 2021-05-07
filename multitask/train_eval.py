import math
import time
import torch
import torch.nn as nn

from utils import format_time

def multi_task_train(args, model, class_h, summarizer_dataloader, classifier_dataloader,
                        device, optimizer, scheduler):
    print("")
    print('Training...')

    criterion = nn.BCEWithLogitsLoss()

    t0 = time.time()

    model.train()
    # Reset the total loss for this epoch.
    total_train_loss1 = 0
    total_train_loss2 = 0

    step = 0
    for batch1, batch2 in zip(summarizer_dataloader, classifier_dataloader):
        # Progress update every 100 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress
            print(' Batch {:>5,} of {:>5,}.   Elapsed: {:}.'.format(step, len(summarizer_dataloader)*2, elapsed))

        for i in range(2):
            model.zero_grad()
            if i == 0:
                # Do summarizer
                loss1 = model(input_ids = batch1[0].to(device),
                             attention_mask = batch1[1].to(device),
                             decoder_input_ids = batch1[2].to(device),
                             decoder_attention_mask = batch1[3].to(device),
                             is_training = True,
                             summarization=True)
                #print('training', loss)

                #import IPython; IPython.embed(); exit(1)

                total_train_loss1 += loss1.mean().detach().cpu()

                # Perform a backward pass to calculate the gradients.
                loss1.backward()

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Update parameters
                optimizer.step()

                # Update the learning rate
                scheduler.step()

            else:
                # Do classification
                logits = model(input_ids = batch2[0].to(device),
                            attention_mask = batch2[1].to(device),
                            decoder_input_ids = batch2[2].to(device),
                            decoder_attention_mask = batch2[3].to(device),
                            labels = batch2[4].to(device),
                            is_training = True)
                _, loss2 = class_h(logits, batch2[4].to(device))
                """
                pred = pred.squeeze()
                pred = pred.unsqueeze(dim=0)
                #import IPython; IPython.embed(); exit(1)

                act_fcn = nn.Sigmoid()
                criterion = nn.BCELoss()
                labels = batch2[4].to(device)
                loss2 = criterion(act_fcn(pred), labels.float())
                """
                total_train_loss2 += loss2.mean().detach().cpu()

                # Perform a backward pass to calculate the gradients.
                loss2.backward()

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Update parameters
                optimizer.step()

                # Update the learning rate
                scheduler.step()

            step += 1

    avg_train_loss1 = total_train_loss1 / len(summarizer_dataloader)
    avg_train_loss2 = total_train_loss2 / len(classifier_dataloader)

    # Measure how long the validation run took.
    training_time = format_time(time.time() - t0)

    return avg_train_loss1, avg_train_loss2, training_time


def multi_task_eval(model, class_h, summarizer_dataloader, classifier_dataloader, device):
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    # Tracking variables
    total_eval_loss1 = 0
    total_eval_loss2 = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch1, batch2 in zip(summarizer_dataloader, classifier_dataloader):
        for i in range(2):
            model.zero_grad()
            if i == 0:
                # Do summarizer
                with torch.no_grad():
                    val_loss1 = model(input_ids = batch1[0].to(device),
                                        attention_mask = batch1[1].to(device),
                                        decoder_input_ids = batch1[2].to(device),
                                        decoder_attention_mask = batch1[3].to(device),
                                        is_training = True,
                                        summarization=True)


                    # Accumulate the validation loss.
                    #print('val_loss', val_loss)
                    total_eval_loss1 += val_loss1.mean().detach().cpu()

            else:
                # Do classifier
                with torch.no_grad():
                    logits = model(input_ids = batch2[0].to(device),
                                        attention_mask = batch2[1].to(device),
                                        decoder_input_ids = batch2[2].to(device),
                                        decoder_attention_mask = batch2[3].to(device),
                                        labels = batch2[4].to(device),
                                        is_training = True)

                    _, val_loss2 = class_h(logits, batch2[4].to(device))
                    #pred = pred.squeeze()
                    #pred = pred.unsqueeze(dim=0)
                    #act_fcn = nn.Sigmoid()
                    #criterion = nn.BCELoss()
                    #val_loss2 = criterion(act_fcn(pred), batch2[4].float())

                    # Accumulate the validation loss.
                    #print('val_loss', val_loss)
                    total_eval_loss2 += val_loss2.mean().detach().cpu()

    # Calculate the average loss over all the batches.
    avg_val_loss1 = total_eval_loss1 / len(summarizer_dataloader)
    avg_val_loss2 = total_eval_loss2 / len(classifier_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    return avg_val_loss1, avg_val_loss2, validation_time
