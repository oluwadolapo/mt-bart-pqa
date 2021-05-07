import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForSequenceClassification, BartConfig

class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout, 
                    pad_idx, hidden_dim, n_layers, bidirectional):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #hidden = [batch size, hid dim * num directions]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        logits = self.fc(hidden)
        act_fcn = nn.Sigmoid()
        #import IPython; IPython.embed(); exit(1)
        return act_fcn(logits)


class lstm_attention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout, 
                    pad_idx, hidden_dim, n_layers, bidirectional):
        super(lstm_attention, self).__init__()
		
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                                bidirectional=bidirectional, dropout=dropout)
        self.label = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
		#self.attn_fc_layer = nn.Linear()
        
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        attn_output = self.attention_net(output, hidden)
        attn_output = self.dropout(attn_output)
        logits = self.label(attn_output)
        act_fcn = nn.Sigmoid()
        #import IPython; IPython.embed(); exit(1)
        return act_fcn(logits)


class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        logits = self.fc(cat)
        act_fcn = nn.Sigmoid()
        #import IPython; IPython.embed(); exit(1)
        return act_fcn(logits)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        logits = self.fc(cat)
        act_fcn = nn.Sigmoid()
        #import IPython; IPython.embed(); exit(1)
        return act_fcn(logits)


class classifier_h(nn.Module):
    def __init__(self):
        super(classifier_h, self).__init__()
        self.fc_head = nn.Linear(1024, 1)
    
    def forward(self, logits, labels):
        classifier_output = self.fc_head(logits)
        classifier_output = classifier_output.squeeze()
        classifier_output = classifier_output.unsqueeze(dim=0)
        #import IPython; IPython.embed(); exit(1)
        act_fcn = nn.Sigmoid()
        criterion = nn.BCELoss()
        #import IPython; IPython.embed(); exit(1)
        pred = act_fcn(classifier_output[0])
        loss = criterion(pred, labels.float())
        return pred, loss

class MyBart(BartForSequenceClassification):
    """
    def classifier_head(self, input_size, device):
        self.fc_head = nn.Linear(input_size, 1)
        self.fc_head.to(device)
    """

    def forward(self, input_ids, labels, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False, device = None):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            return_dict=True
        )

        logits = outputs['last_hidden_state'][:,-1,:]
        """
        #import IPython; IPython.embed(); exit(1)
        class_head_size = outputs['last_hidden_state'].size()[2]
        self.classifier_head(class_head_size, device)
        classifier_output = self.fc_head(logits)
        classifier_output = classifier_output.squeeze()
        classifier_output = classifier_output.unsqueeze(dim=0)
        #import IPython; IPython.embed(); exit(1)
        act_fcn = nn.Sigmoid()
        criterion = nn.BCELoss()
        #import IPython; IPython.embed(); exit(1)
        loss = criterion(act_fcn(classifier_output[0]), labels.float())
        return act_fcn(classifier_output[0]), loss
        """
        return logits


def model_choice(bart_type, from_scratch, model_path):
    if from_scratch:
        model = MyBart.from_pretrained(bart_type)
        tokenizer = BartTokenizer.from_pretrained(bart_type)
    else:
        model = MyBart.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
    return model, tokenizer