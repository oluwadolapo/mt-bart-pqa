import wandb
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartForSequenceClassification, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import format_time, flat_accuracy, joint_metrics, confusion
from multitask import config
from multitask.model import model_choice, classifier_h
from multitask.train_eval import multi_task_train, multi_task_eval
from multitask.data import generator_data, verifier_data
from multitask.data import verifier_testing_data



class BartGenerator:
  def __init__(self, output_dir, device):
    self.model = BartForConditionalGeneration.from_pretrained(output_dir)
    self.model.to(device)
    self.device = device
    self.tokenizer = BartTokenizer.from_pretrained(output_dir)

  def encode(self,question_context):
    encoded = self.tokenizer.encode_plus(question_context, max_length = 512, truncation=True, return_tensors='pt')
    return encoded["input_ids"], encoded["attention_mask"]

  def predict(self,question_context):
    input_ids, attention_mask = self.encode(question_context)
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    summary_ids = self.model.generate(input_ids, attention_mask = attention_mask, num_beams=4, max_length=50)
    summary = [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in summary_ids]
    return summary



def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_generator_test_data(df):
    questions = [q if q.endswith("?") else q+"?" for q in df.question]
    reviews = [r for r in df.passages]
    #answers = [a for a in df.answers]
    multiple_answers = [a for a in df.multiple_answers]

    questions = [q.lower() + " </s>" for q in questions]
    reviews = [[p.lower() for p in r] for r in reviews]
    #answers = ["<s> " + a.lower() + " </s>" for a in answers]
    multiple_answers = [["<s> " + a.lower() + " </s>" for a in ans] for ans in multiple_answers]

    quest_rev = []
    for i, q in enumerate(questions):
        for r in reviews[i]:
            q = q + " " + r
        quest_rev.append(q)

    return quest_rev, multiple_answers


def optim(args, model, train_dataloader):
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


def training(args, model, class_h, summarizer_train_dataloader, summarizer_val_dataloader,
                classifier_train_dataloader, classifier_val_dataloader, tokenizer, device):
    training_stats = []
    total_t0 = time.time()
    min_val_loss1 = None
    min_val_loss2 = None

    optimizer, scheduler = optim(args, model, summarizer_train_dataloader)

    for epoch in range(1, args.num_train_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))

        
        avg_train_loss1, avg_train_loss2, training_time = multi_task_train(args, model, class_h, summarizer_train_dataloader, 
                                          classifier_train_dataloader, device, optimizer, scheduler)
        avg_val_loss1, avg_val_loss2, validation_time = multi_task_eval(model, class_h, summarizer_val_dataloader,
                                                         classifier_val_dataloader, device)
        
        wandb.log({"avg_train_loss1": avg_train_loss1, "avg_train_loss2": avg_train_loss2, "avg_val_loss1": avg_val_loss1, "avg_val_loss2": avg_val_loss2}, step=epoch)
 
        print()
        print("   Average training loss for summarizer: {0:.2f}".format(avg_train_loss1))
        print("   Average validation loss for summarizer: {0:.2f}".format(avg_val_loss1))
        print()
        print("   Average training loss for classifier: {0:.2f}".format(avg_train_loss2))
        print("   Average validation loss for classifier: {0:.2f}".format(avg_val_loss2))
        print()
        print("   Training epoch took: {:}".format(training_time))
 
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss1': avg_train_loss1,
                'Valid. Loss1': avg_val_loss1,
                'Training Loss2': avg_train_loss2,
                'Valid. Loss2': avg_val_loss2,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
    
        if min_val_loss1 == None or avg_val_loss1 < min_val_loss1:
            if min_val_loss2 == None or avg_val_loss2 < min_val_loss2:
                print("Saving model to %s" % args.save_bart_path)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.save_bart_path)
                tokenizer.save_pretrained(args.save_bart_path)
                torch.save(class_h.state_dict(), args.save_classifier_path)
                # Copy the model files to a directory in your Google Drive.
                #!cp -r ./model_save4/ "/content/drive/My Drive/Experiments/Bart_QA"
 
                min_val_loss1 = avg_val_loss1
                min_val_loss2 = avg_val_loss2
 
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
 
    print("")
    print("Training complete!")
 
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

def classification_eval(model, class_h, test_dataloader, device):
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

    if args.exp_type == "train":
        model, tokenizer = model_choice(args.bart_type, args.from_scratch, args.load_bart_path)
        model.to(device)

        class_h = classifier_h()
        if not args.from_scratch:
            class_h.load_state_dict(torch.load(args.load_classifier_path, map_location=device))
        class_h.to(device)

        wandb.watch(model)
        wandb.watch(class_h)

        summarizer_train_dataloader, summarizer_val_dataloader = generator_data(args, tokenizer, 'summarization')
        classifier_train_dataloader, classifier_val_dataloader = verifier_data(args, tokenizer, 'classification')

        training(args, model, class_h, summarizer_train_dataloader, summarizer_val_dataloader,
                classifier_train_dataloader, classifier_val_dataloader, tokenizer, device)

    elif args.exp_type == "test_classifier":
        model, tokenizer = model_choice(args.bart_type, args.from_scratch, args.load_bart_path)
        model.to(device)
        class_h = classifier_h()
        class_h.load_state_dict(torch.load(args.load_classifier_path, map_location=device))
        class_h.to(device)

        wandb.watch(model)
        wandb.watch(class_h)

        test_dataloader = verifier_testing_data(args, tokenizer)

        model.eval()
        class_h.eval()
        classification_eval(model, class_h, test_dataloader, device)

    elif args.exp_type == "test_generator":
        model = BartGenerator(args.load_bart_path, device)
        df = pd.read_json(args.generator_data_path, orient='split')
        df = df.head(5)
        #df = df[:1]
        quest_rev, ref_answers = prepare_generator_test_data(df)
        print()
        print('Testing the model')
        pred_answers = []
        for i in range(len(quest_rev)):
            answer = model.predict(quest_rev[i])[0]
            pred_answers.append(answer)
    
        questions = [q.lower() if q.endswith("?") else q.lower()+"?" for q in df.question]
        multiple_answers = [[a.lower() for a in ans] for ans in df.multiple_answers]
        print()
        print('Saving the outputs')

        output_json = {'Question':questions,
                    'Pred_answer':pred_answers,
                    'Ref_answers':multiple_answers}

        with open(args.output_json_path, 'w') as json_file:
            json.dump(output_json, json_file)
    
        print()
        print('Done')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')

