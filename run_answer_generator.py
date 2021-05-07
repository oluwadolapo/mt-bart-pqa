import wandb

import time
import random
import json

import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup

from generator.model import model_choice
from generator import config
from generator.train_eval import train, eval
from generator.data import training_data
from utils import format_time

class BartGenerator:
  def __init__(self, bart_path, device):
    self.model = BartForConditionalGeneration.from_pretrained(bart_path)
    self.model.to(device)
    self.device = device
    self.tokenizer = BartTokenizer.from_pretrained(bart_path)

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


def prepare_test_data(df):
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

def training(args, model, train_dataloader, val_dataloader, tokenizer, device):
    training_stats = []
    total_t0 = time.time()
    min_val_loss = None

    optimizer, scheduler = optim(args, model, train_dataloader)

    for epoch in range(1, args.num_train_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))
 
        avg_train_loss, avg_train_perplexity, training_time = train(args, model, train_dataloader, 
                                                                device, optimizer, scheduler)
        avg_val_loss, avg_val_perplexity, validation_time = eval(model, val_dataloader, device)
        
        wandb.log({"avg_train_loss": avg_train_loss, "avg_train_perplexity": avg_train_perplexity, "avg_val_loss": avg_val_loss, "avg_val_perplexity": avg_val_perplexity}, step=epoch)
 
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
                'Training Ppl.': avg_train_perplexity,
                'Valid. Ppl.': avg_val_perplexity,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
    
        if min_val_loss == None or avg_val_loss < min_val_loss:
            print("Saving model to %s" % args.save_model_path)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.save_model_path)
            tokenizer.save_pretrained(args.save_model_path)
 
            # Copy the model files to a directory in your Google Drive.
            #!cp -r ./model_save4/ "/content/drive/My Drive/Experiments/Bart_QA"
 
            min_val_loss = avg_val_loss
 
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
 
    print("")
    print("Training complete!")
 
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


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

    if args.exp_type == "train_bart":
        model, tokenizer = model_choice(args.bart_type, args.from_scratch, args.load_model_path)
        model.to(device)

        wandb.watch(model)

        train_dataloader, val_dataloader = training_data(args, tokenizer)
        training(args, model, train_dataloader, val_dataloader, tokenizer, device)
        print()
        print("End of Training")

    elif args.exp_type == "test_bart":
        model = BartGenerator(args.load_model_path, device)
        df = pd.read_json(args.data_path, orient='split')
        #df = df.head(1)
        #df = df[:1]
        quest_rev, ref_answers = prepare_test_data(df)
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
        print('Done with testing')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')

