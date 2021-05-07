import random
import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize
import flask
from flask import Flask, request, render_template
import json

app = Flask(__name__)

class SummarizerModel:
  def __init__(self, output_dir):
    self.model = BartForConditionalGeneration.from_pretrained(output_dir)
    self.tokenizer = BartTokenizer.from_pretrained(output_dir)

  def encode(self,question_context):
    encoded = self.tokenizer.encode_plus(question_context, max_length = 512, truncation=True, return_tensors='pt')
    return encoded["input_ids"], encoded["attention_mask"]

  def predict(self,question_context):
    input_ids, attention_mask = self.encode(question_context)
    summary_ids = self.model.generate(input_ids, attention_mask = attention_mask, num_beams=4, max_length=50)
    summary = [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in summary_ids]
    return summary

@app.route('/')
def index():
    return render_template('index.html')

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        _set_random_seeds(42)
        q_id = request.json['input_idx']
        q_id = int(q_id)

        try:
            df = pd.read_json('/home/oluwadolapo/Datasets/amazonQA_test.json', orient='split')
            question_text = df["question"][q_id]
            reviews = ""
            for rev in df["passages"][q_id]:
                reviews += rev + " "

            question_text = question_text.lower()
            reviews = reviews.lower()

            if not question_text.endswith("?"):
                question_text += "?"

            with open('outputs/bm25.json', 'r') as json_file:
                pred_json = json.load(json_file)
            bm25_response = pred_json['Pred_answer'][q_id].split(".")[0].strip() + "."

            with open('outputs/tfidf.json', 'r') as json_file:
                pred_json = json.load(json_file)
            tfidf_response = pred_json['Pred_answer'][q_id].split(".")[0].strip() + "."
        
            with open('outputs/d2v.json', 'r') as json_file:
                pred_json = json.load(json_file)
            doc2vec_response = pred_json['Pred_answer'][q_id].split(".")[0].strip() + "."

            with open('outputs/bert.json', 'r') as json_file:
                pred_json = json.load(json_file)
            sbert_response = pred_json['Pred_answer'][q_id].split(".")[0].strip() + "."

            with open('outputs/bart.json', 'r') as json_file:
                pred_json = json.load(json_file)
            stbart_response = pred_json['Pred_answer'][q_id].split(".")[0].strip() + "."

            with open('outputs/mt_bart.json', 'r') as json_file:
                pred_json = json.load(json_file)
            mtbart_response = pred_json['Pred_answer'][q_id].split(".")[0].strip() + "."

            print(str(tfidf_response))

            res = {'question': question_text,
                    'reviews': reviews,
                    'bm25': bm25_response,
                    'tf-idf': tfidf_response,
                    'doc2vec': doc2vec_response,
                    's-bert': sbert_response,
                    'st-bart': stbart_response,
                    'mt-bart': mtbart_response}

            return flask.jsonify(res)

        except KeyError:
            return app.response_class(response=json.dumps("Index out of range. Enter a lower question index"), status=500, mimetype='application/json')

    except Exception as error:
        res = str(error)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8002, use_reloader=True)
