import json
import pandas as pd
import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', dest='output_path', type=str, default=None,
                        help='Directory for the json output path')
    args = parser.parse_args()
    return args

def main():
    args = get_params()
    with open(args.output_path, 'r') as json_file:
        pred_json = json.load(json_file)

    questions = pred_json['Question']
    pred_answers = pred_json['Pred_answer']
    ref_answers = pred_json['Ref_answers']

    idx = int(input("Enter index of sample to display ===> "))
    print()
    print(questions[idx])
    print()
    print(pred_answers[idx].split(".")[0].strip() + ".")
    print()
    print(ref_answers[idx])

if __name__ == "__main__":
    main()