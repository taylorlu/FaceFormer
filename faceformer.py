import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=30, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Faceformer(nn.Module):
    def __init__(self, args, speaker_len):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        exp_jaw: (batch_size, seq_len, 53)
        """
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        # motion encoder
        self.exp_jaw_map = nn.Linear(args.exp_jaw_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.exp_jaw_map_r = nn.Linear(args.feature_dim, args.exp_jaw_dim)
        # style embedding
        self.obj_vector = nn.Linear(speaker_len, args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.exp_jaw_map_r.weight, 0)
        nn.init.constant_(self.exp_jaw_map_r.bias, 0)

    def forward(self, audio, exp_jaw, one_hot, criterion, teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        frame_num = exp_jaw.shape[1]
        hidden_states = self.audio_encoder(audio, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)

        if teacher_forcing:
            exp_jaw_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = exp_jaw_emb  
            template = torch.zeros([1, exp_jaw.shape[1], exp_jaw.shape[2]])
            exp_jaw_input = torch.cat((template, exp_jaw[:,:-1]), 1) # shift one position
            exp_jaw_input = self.exp_jaw_map(exp_jaw_input)
            exp_jaw_input = exp_jaw_input + style_emb
            exp_jaw_input = self.PPE(exp_jaw_input)
            tgt_mask = self.biased_mask[:, :exp_jaw_input.shape[1], :exp_jaw_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, exp_jaw_input.shape[1], hidden_states.shape[1])
            exp_jaw_out = self.transformer_decoder(exp_jaw_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            exp_jaw_out = self.exp_jaw_map_r(exp_jaw_out)
        else:
            for i in range(frame_num):
                if i==0:
                    exp_jaw_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = exp_jaw_emb
                    exp_jaw_input = self.PPE(style_emb)
                else:
                    exp_jaw_input = self.PPE(exp_jaw_emb)
                tgt_mask = self.biased_mask[:, :exp_jaw_input.shape[1], :exp_jaw_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, exp_jaw_input.shape[1], hidden_states.shape[1])
                exp_jaw_out = self.transformer_decoder(exp_jaw_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                exp_jaw_out = self.exp_jaw_map_r(exp_jaw_out)
                new_output = self.exp_jaw_map(exp_jaw_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                exp_jaw_emb = torch.cat((exp_jaw_emb, new_output), 1)

        loss = criterion(exp_jaw_out, exp_jaw) # (batch, seq_len, 53)
        loss = torch.mean(loss)
        return loss

    def predict(self, audio, one_hot):
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio).last_hidden_state
        frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i==0:
                exp_jaw_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = exp_jaw_emb
                exp_jaw_input = self.PPE(style_emb)
            else:
                exp_jaw_input = self.PPE(exp_jaw_emb)

            tgt_mask = self.biased_mask[:, :exp_jaw_input.shape[1], :exp_jaw_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, exp_jaw_input.shape[1], hidden_states.shape[1])
            exp_jaw_out = self.transformer_decoder(exp_jaw_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            exp_jaw_out = self.vertice_map_r(exp_jaw_out)
            new_output = self.vertice_map(exp_jaw_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            exp_jaw_emb = torch.cat((exp_jaw_emb, new_output), 1)

        return exp_jaw_out
