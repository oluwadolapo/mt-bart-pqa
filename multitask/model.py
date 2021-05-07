import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class classifier_h(nn.Module):
    def __init__(self):
        super(classifier_h, self).__init__()
        self.fc_hidden = nn.Linear(1024, 1024)
        self.fc_head = nn.Linear(1024, 1)
    
    def forward(self, logits, labels):
        hidden_vec = self.fc_hidden(logits)
        hidden_vec = self.fc_hidden(hidden_vec)
        classifier_output = self.fc_head(hidden_vec)
        classifier_output = classifier_output.squeeze()
        classifier_output = classifier_output.unsqueeze(dim=0)
        #import IPython; IPython.embed(); exit(1)
        act_fcn = nn.Sigmoid()
        criterion = nn.BCELoss()
        #import IPython; IPython.embed(); exit(1)
        pred = act_fcn(classifier_output[0])
        loss = criterion(pred, labels.float())
        return pred, loss

class MyBart(BartForConditionalGeneration):
    def classifier_head(self, voc_size, device):
        self.fc_head = nn.Linear(voc_size, 1)
        self.fc_head.to(device)

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False, summarization=False, labels=None, device=None):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()
        if summarization:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache
            )
            lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            if is_training:
                loss_fct = nn.CrossEntropyLoss(reduce=False)
                losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                                  decoder_input_ids.view(-1))
                loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
                return loss
            return (lm_logits, ) + outputs[1:]
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
                use_cache=use_cache,
                return_dict=True
            )
            logits = outputs['last_hidden_state'][:,-1,:]
            return logits


def model_choice(bart_type, from_scratch, model_path):
    if from_scratch:
        model = MyBart.from_pretrained(bart_type)
        tokenizer = BartTokenizer.from_pretrained(bart_type)
    else:
        model = MyBart.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
    
    return model, tokenizer