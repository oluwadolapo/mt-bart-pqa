import wandb

from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BartTokenizer, BartForSequenceClassification, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

import random
import time

from verifier import config
from verifier.train import *
from verifier.model import *
from verifier.data import *
from utils import format_time
from utils import flat_accuracy, joint_metrics, confusion


def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_lstm_cnn(args, model, test_iterator, device):
    print("")
    print('Running Evaluation...')

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
    
    for batch in test_iterator:
        if args.model_type == "lstm":
            text, text_lengths = batch.text
            with torch.no_grad():
                predictions = model(text, text_lengths).squeeze(1)
        elif args.model_type == "cnn":    
            with torch.no_grad():
                predictions = model(batch.text).squeeze(1)

        # Move predictions and labels to CPU
        predictions = predictions.to(dtype=int).detach().cpu().numpy()
        labels = batch.label.to(dtype=int).detach().cpu().numpy()
    
        # Calculate the metrics for this batch of test sentences, and
        # accumulate it over all batches
        total_test_accuracy += flat_accuracy(predictions, labels)
    
        precision, recall, f1 = joint_metrics(predictions, labels)
    
        total_test_precision += precision
        total_test_recall += recall
        total_test_f1 += f1

        TN, FP, FN, TP = confusion(predictions, labels)
        total_TN += TN
        total_FP += FP
        total_FN += FN
        total_TP += TP
    #import IPython; IPython.embed(); exit(1)
    # Report the final metrics for this validation run.
    avg_test_accuracy = total_test_accuracy / len(test_iterator)
    avg_test_precision = total_test_precision / len(test_iterator)
    avg_test_recall = total_test_recall / len(test_iterator)
    avg_test_f1 = total_test_f1 / len(test_iterator)

    wandb.run.summary["test_accuracy"] = avg_test_accuracy
    wandb.run.summary["test_precision"] = avg_test_precision
    wandb.run.summary["test_recall"] = avg_test_recall
    wandb.run.summary["test_f1"] = avg_test_f1
    wandb.run.summary["TN"] = total_TN
    wandb.run.summary["FP"] = total_FP
    wandb.run.summary["FN"] = total_FN
    wandb.run.summary["TP"] = total_TP
    wandb.run.summary.update()

    print(" Accuracy:  {0:.2f}".format(avg_test_accuracy))
    print(" Precision: {0:.2f}".format(avg_test_precision))
    print(" Recall:    {0:.2f}".format(avg_test_recall))
    print(" F1:        {0:.2f}".format(avg_test_f1))

    print()

    print(" TN:  {0:.2f}".format(total_TN))
    print(" FP:  {0:.2f}".format(total_FP))
    print(" FN:  {0:.2f}".format(total_FN))
    print(" TP:  {0:.2f}".format(total_TP))


def evaluate_bart(model, class_h, test_dataloader, device):
    print("")
    print("Running Evaluation...")
    
    # Tracking variables
    total_test_accuracy = 0
    total_test_precision = 0
    total_test_recall = 0
    total_test_f1 = 0

    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    # Evaluate data for one epoch
    for batch in test_dataloader:

        with torch.no_grad():
            logits = model(input_ids = batch[0].to(device),
                            attention_mask = batch[1].to(device),
                            decoder_input_ids = batch[2].to(device),
                            decoder_attention_mask = batch[3].to(device),
                            labels = batch[4].to(device),
                            device = device)

            pred, _ = class_h(logits, batch[4].to(device))

        b_labels = batch[4].to(device)
        # Move logits and labels to CPU
        pred = pred.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        #pred_flat = np.argmax(logits, axis=1).flatten()
        pred_flat = pred.flatten()
        pred_flat = np.round_(pred_flat)
        labels_flat = label_ids.flatten()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches
        total_test_accuracy += flat_accuracy(pred_flat, labels_flat)
    
        precision, recall, f1 = joint_metrics(pred_flat, labels_flat)
    
        total_test_precision += precision
        total_test_recall += recall
        total_test_f1 += f1

        TN, FP, FN, TP = confusion(pred_flat, labels_flat)
        total_TN += TN
        total_FP += FP
        total_FN += FN
        total_TP += TP

    # Report the final accuracy for this validation run.
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    avg_test_precision = total_test_precision / len(test_dataloader)
    avg_test_recall = total_test_recall / len(test_dataloader)
    avg_test_f1 = total_test_f1 / len(test_dataloader)

    wandb.run.summary["test_accuracy"] = avg_test_accuracy
    wandb.run.summary["test_precision"] = avg_test_precision
    wandb.run.summary["test_recall"] = avg_test_recall
    wandb.run.summary["test_f1"] = avg_test_f1
    wandb.run.summary["TN"] = total_TN
    wandb.run.summary["FP"] = total_FP
    wandb.run.summary["FN"] = total_FN
    wandb.run.summary["TP"] = total_TP
    wandb.run.summary.update()

    print(" Accuracy:  {0:.2f}".format(avg_test_accuracy))
    print(" Precision: {0:.2f}".format(avg_test_precision))
    print(" Recall:    {0:.2f}".format(avg_test_recall))
    print(" F1:        {0:.2f}".format(avg_test_f1))

    print()

    print(" TN:  {0:.2f}".format(total_TN))
    print(" FP:  {0:.2f}".format(total_FP))
    print(" FN:  {0:.2f}".format(total_FN))
    print(" TP:  {0:.2f}".format(total_TP))


def lstm_cnn_optim(model):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, criterion

def bart_optim(args, model, train_dataloader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    total_steps = len(train_dataloader) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                num_warmup_steps=args.warmup_steps,
                                num_training_steps=total_steps)
    return optimizer, scheduler

def lstm_cnn_training(args, model, vocab, device, *data_iterators):
    optimizer, criterion = lstm_cnn_optim(model)
    criterion = criterion.to(device)
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    total_t0 = time.time()
    for epoch in range(1, args.num_train_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))

        train_loss, train_acc, train_time = train_lstm_cnn(args, model, data_iterators[0], optimizer, criterion)
        valid_loss, valid_acc, valid_time = validate_lstm_cnn(args, model, data_iterators[1], criterion)

        wandb.log({"avg_train_loss": train_loss, "avg_train_acc": train_acc, "avg_val_loss": valid_loss, "avg_val_acc": valid_acc}, step=epoch)
        
        print("")
        print("   Average training loss: {0:.2f}".format(train_loss))
        print("   Training epoch took: {:}".format(train_time))

        print("   Validation Loss: {0:.2f}".format(valid_loss))
        print("   Validation took: {:}".format(valid_time))

        #if valid_loss < best_valid_loss:
        if best_valid_acc < valid_acc:
            print("Saving model to %s" % args.save_model_path)
            wandb.run.summary["best_val_loss"] = valid_loss
            wandb.run.summary["best-loss-epoch"] = epoch
            #best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), args.save_model_path)

            # Save embeddings
            print("")
            print("Saving embeddings to %s" % args.save_emb_path)
            embeddings = model.embedding.weight.data
            with open(args.save_emb_path, 'w') as f:
                for i, embedding in enumerate(tqdm(embeddings)):
                    word = vocab.itos[i]
                    #skip words with unicode symbols
                    if len(word) != len(word.encode()) | len(embedding) != args.emb_dim:
                        continue
                    vector = ' '.join([str(i) for i in embedding.tolist()])
                    f.write(f'{word} {vector}\n')
            # Test the model
            test_iterator, _ = test_lstm_cnn_data(args, device)
            evaluate_lstm_cnn(args, model, test_iterator, device)


def bart_training(args, model, class_h, train_dataloader, val_dataloader, tokenizer, device):
    training_stats = []
    total_t0 = time.time()
    min_val_loss = None

    optimizer, scheduler = bart_optim(args, model, train_dataloader)

    for epoch in range(1, args.num_train_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))
 
        avg_train_loss, training_time = train_bart(args, model, class_h, train_dataloader, 
                                                                device, optimizer, scheduler)
        avg_val_loss, validation_time = validate_bart(model, class_h, val_dataloader, device)
        
        wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss}, step=epoch)
 
        print("")
        print("   Average training loss: {0:.2f}".format(avg_train_loss))
        print("   Average validation loss: {0:.2f}".format(avg_val_loss))
        print("   Training epoch took: {:}".format(training_time))
 
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
    
        if min_val_loss == None or avg_val_loss < min_val_loss:
            print("Saving model to %s" % args.save_bart_path)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.save_bart_path)
            tokenizer.save_pretrained(args.save_bart_path)
            torch.save(class_h.state_dict(), args.save_classifier_path)
 
            # Copy the model files to a directory in your Google Drive.
            #!cp -r ./model_save4/ "/content/drive/My Drive/Experiments/Bart_QA"
 
            min_val_loss = avg_val_loss
 
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
 
    print("")
    print("Training complete!")
 
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


def model_choice(args, vocab=None, UNK_IDX=None, PAD_IDX=None, device=None):
    if args.model_type == 'lstm':
        INPUT_DIM = len(vocab)
        EMBEDDING_DIM = args.emb_dim
        OUTPUT_DIM = 1
        DROPOUT = args.dropout
        HIDDEN_DIM = args.lstm_hid_dim
        N_LAYERS = args.n_lstm_layers
        BIDIRECTIONAL = True

        if args.with_attention:
            model = lstm_attention(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, DROPOUT, PAD_IDX, 
                        HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL)
        else:
            model = lstm(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, DROPOUT, PAD_IDX, 
                        HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL)

        if args.from_scratch:
            #Initialize the pretrained embedding
            pretrained_embeddings = vocab.vectors
            model.embedding.weight.data.copy_(pretrained_embeddings)

            #zero the initial weights of the unknown and padding tokens.
            EMBEDDING_DIM = args.emb_dim
            model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        else:
            model.load_state_dict(torch.load(args.load_model_path))
        return model

    elif args.model_type == 'cnn':
        INPUT_DIM = len(vocab)
        EMBEDDING_DIM = args.emb_dim
        N_FILTERS = args.n_filters
        FILTER_SIZES = [int(s) for s in args.filter_sizes]
        OUTPUT_DIM = 1
        DROPOUT = args.dropout
        model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                    OUTPUT_DIM, DROPOUT, PAD_IDX)
        if args.from_scratch:
            #Initialize the pretrained embedding
            pretrained_embeddings = vocab.vectors
            model.embedding.weight.data.copy_(pretrained_embeddings)

            #zero the initial weights of the unknown and padding tokens.
            EMBEDDING_DIM = args.emb_dim
            model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        else:
            model.load_state_dict(torch.load(args.load_model_path))
        return model

    elif args.model_type == 'bart':
        class_h = classifier_h()
        if args.from_scratch:
            model = MyBart.from_pretrained(args.bart_type)
            tokenizer = BartTokenizer.from_pretrained(args.bart_type)
        else:
            # import IPython; IPython.embed(); exit(1)
            model = MyBart.from_pretrained(args.load_bart_path)
            class_h.load_state_dict(torch.load(args.load_classifier_path, map_location=device))
            tokenizer = BartTokenizer.from_pretrained(args.load_bart_path)
        return model, class_h, tokenizer


def main():
    args = config.get_params()
    wandb.init(config=args, project=args.project_name)

    _set_random_seeds(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    if args.model_type == "lstm" or args.model_type == "cnn":
        train_iterator, valid_iterator, vocab, UNK_IDX, PAD_IDX = train_lstm_cnn_data(args, device)
        model = model_choice(args, vocab, UNK_IDX, PAD_IDX)
        model = model.to(device)
        wandb.watch(model)
        lstm_cnn_training(args, model, vocab, device, 
                train_iterator, valid_iterator)
        print()
        print("End of Training")

    elif args.model_type == "bart" and args.exp_type == "train_bart":
        model, class_h, tokenizer = model_choice(args, device = device)
        model.to(device)
        class_h.to(device)
        wandb.watch(model)
        wandb.watch(class_h)
        train_dataloader, val_dataloader = bart_training_data(args, tokenizer)
        bart_training(args, model, class_h, train_dataloader, val_dataloader, tokenizer, device)
        print()
        print("End of Training")
    elif args.model_type == "bart" and args.exp_type == "test_bart":
        model, class_h, tokenizer = model_choice(args, device = device)
        model.to(device)
        class_h.to(device)
        wandb.watch(model)
        wandb.watch(class_h)
        test_dataloader = bart_testing_data(args, tokenizer)
        model.eval()
        evaluate_bart(model, class_h, test_dataloader, device)
        print()
        print("End of Testing")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')