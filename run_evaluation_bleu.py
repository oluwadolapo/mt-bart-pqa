import nltk
from nltk.translate.bleu_score import sentence_bleu
import argparse
import json
#nltk.download('punkt')

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for the json data path')
    args = parser.parse_args()
    return args

#review = nltk.word_tokenize(review)
def main():
    args = get_params()

    with open(args.data_path, 'r') as json_file:
        pred_json = json.load(json_file)

    questions = pred_json['Question']
    pred_answers = pred_json['Pred_answer']
    ref_answers = pred_json['Ref_answers']
    reference = [[nltk.word_tokenize(item) for item in answers] for answers in ref_answers]
    candidate = [nltk.word_tokenize(pred_answer.split(".")[0].strip() + ".") for pred_answer in pred_answers]
    #candidate = [nltk.word_tokenize(pred_answer) for pred_answer in pred_answers]
    
    #reference = [['this', 'is', 'a', 'test']]
    #candidate = ['this', 'is', 'a', 'test']
    # print('Individual 1-gram: %f' % sentence_bleu(reference[3], candidate[3], weights=(1, 0, 0, 0)))
    # print('Individual 2-gram: %f' % sentence_bleu(reference[3], candidate[3], weights=(0, 1, 0, 0)))
    # print('Individual 3-gram: %f' % sentence_bleu(reference[3], candidate[3], weights=(0, 0, 1, 0)))
    #print('Individual 4-gram: %f' % sentence_bleu(reference[2], candidate[2], weights=(0, 0, 0, 1)))
    #import IPython; IPython.embed(); exit(1);

    all_bleu1 = []
    #all_bleu2 = []

    for count in range(len(pred_answers)):
        bleu1 = []
        #bleu2 = []

        for ans in reference[count]:
            bleu1.append(sentence_bleu(ans, candidate[count], weights=(1, 0, 0, 0)))
            #bleu2.append(sentence_bleu(ans, candidate[count], weights=(0, 1, 0, 0)))


        bleu1.sort()
        #bleu2.sort()

        all_bleu1.append(bleu1[-1])
        #all_bleu2.append(bleu2[-1])

    print()
    print("BLEU1:", sum(all_bleu1)/len(all_bleu1))
    #print("BLEU2:", sum(all_bleu2)/len(all_bleu2))
    print()
    print('Total no of samples evaluated: ', len(all_bleu1))

if __name__ == "__main__":
    main()
