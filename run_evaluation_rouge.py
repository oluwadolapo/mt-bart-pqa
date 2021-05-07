import pandas as pd
from rouge import Rouge
from rouge_score import rouge_scorer
import argparse
import json

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for the json data path')
    args = parser.parse_args()
    return args

def main():
    args = get_params()

    with open(args.data_path, 'r') as json_file:
        pred_json = json.load(json_file)

    questions = pred_json['Question']
    pred_answers = pred_json['Pred_answer']
    ref_answers = pred_json['Ref_answers']

    #rouge = Rouge()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    all_rouge_1p = []
    all_rouge_1r = []
    all_rouge_1f = []
    all_rouge_2p = []
    all_rouge_2r = []
    all_rouge_2f = []
    all_rouge_lp = []
    all_rouge_lr = []
    all_rouge_lf = []

    for count in range(len(pred_answers)):
        rouge_1p = []
        rouge_1r = []
        rouge_1f = []
        rouge_2p = []
        rouge_2r = []
        rouge_2f = []
        rouge_lp = []
        rouge_lr = []
        rouge_lf = []

        """
        for ans in ref_answers[count]:
            try:
                scores = rouge.get_scores([pred_answers[count]], [ans], avg=True)
                rouge_1p.append(scores['rouge-1']['p'])
                rouge_1r.append(scores['rouge-1']['r'])
                rouge_1f.append(scores['rouge-1']['f'])
                rouge_2p.append(scores['rouge-2']['p'])
                rouge_2r.append(scores['rouge-2']['r'])
                rouge_2f.append(scores['rouge-2']['f'])
                rouge_lp.append(scores['rouge-l']['p'])
                rouge_lr.append(scores['rouge-l']['r'])
                rouge_lf.append(scores['rouge-l']['f'])
            except ValueError:
                continue
        """

        #"""
        for ans in ref_answers[count]:
            scores = scorer.score(pred_answers[count].split(".")[0].strip() + ".", ans)
            rouge_1p.append(scores['rouge1'][0])
            rouge_1r.append(scores['rouge1'][1])
            rouge_1f.append(scores['rouge1'][2])
            rouge_2p.append(scores['rouge2'][0])
            rouge_2r.append(scores['rouge2'][1])
            rouge_2f.append(scores['rouge2'][2])
            rouge_lp.append(scores['rougeL'][0])
            rouge_lr.append(scores['rougeL'][1])
            rouge_lf.append(scores['rougeL'][2])
        #"""


        rouge_1p.sort()
        rouge_1r.sort()
        rouge_1f.sort()
        rouge_2p.sort()
        rouge_2r.sort()
        rouge_2f.sort()
        rouge_lp.sort()
        rouge_lr.sort()
        rouge_lf.sort()

        all_rouge_1p.append(rouge_1p[-1])
        all_rouge_1r.append(rouge_1r[-1])
        all_rouge_1f.append(rouge_1f[-1])
        all_rouge_2p.append(rouge_2p[-1])
        all_rouge_2r.append(rouge_2r[-1])
        all_rouge_2f.append(rouge_2f[-1])
        all_rouge_lp.append(rouge_lp[-1])
        all_rouge_lr.append(rouge_lr[-1])
        all_rouge_lf.append(rouge_lf[-1])

    print()
    print("rouge_1p:", sum(all_rouge_1p)/len(all_rouge_1p))
    print("rouge_1r:", sum(all_rouge_1r)/len(all_rouge_1r))
    print("rouge_1f:", sum(all_rouge_1f)/len(all_rouge_1f))
    print()
    print("rouge_2p:", sum(all_rouge_2p)/len(all_rouge_2p))
    print("rouge_2r:", sum(all_rouge_2r)/len(all_rouge_2r))
    print("rouge_2f:", sum(all_rouge_2f)/len(all_rouge_2f))
    print()
    print("rouge_lp:", sum(all_rouge_lp)/len(all_rouge_lp))
    print("rouge_lr:", sum(all_rouge_lr)/len(all_rouge_lr))
    print("rouge_lf:", sum(all_rouge_lf)/len(all_rouge_lf))

    print()
    print('Total no of samples evaluated: ', len(all_rouge_lf))

if __name__ == "__main__":
    main()
