import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        voc_size = lm_logits.size()[2]
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            return lm_logits, loss
        return (lm_logits, ) + outputs[1:]


def model_choice(bart_type, from_scratch, model_path):
    if from_scratch:
        model = MyBart.from_pretrained(bart_type)
        tokenizer = BartTokenizer.from_pretrained(bart_type)
    else:
        model = MyBart.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
    
    return model, tokenizer