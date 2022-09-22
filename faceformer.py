import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math, os
from wav2vec import Wav2Vec2Model
from flame.FLAME import FLAME

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
def enc_dec_mask(device, dataset, T, S):
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
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        if(args.facebook):
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        else:
            self.audio_encoder = Wav2Vec2Model.from_pretrained("TencentGameMate/chinese-wav2vec2-base")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        # motion encoder
        self.exp_jaw_map = nn.Linear(args.exp_jaw_dim, args.feature_dim)
        self.max_seq_len = args.max_seq_len
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period, max_seq_len=self.max_seq_len)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=self.max_seq_len, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.exp_jaw_map_r = nn.Linear(args.feature_dim, args.exp_jaw_dim)
        # style embedding
        self.obj_vector = nn.Linear(speaker_len, args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.exp_jaw_map_r.weight, 0)
        nn.init.constant_(self.exp_jaw_map_r.bias, 0)

        flame_model_path = os.path.join('models', 'generic_model-2020.pkl') 
        flame_lmk_embedding_path = os.path.join('models', 'flame_static_embedding.pkl') 
        self.flame = FLAME(flame_model_path, flame_lmk_embedding_path, n_shape=300, n_exp=50).to(torch.device(args.device))

    def forward(self, audio, template, vertice, one_hot, keep_mask, teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        frame_num = vertice.shape[1]
        batch_size = vertice.shape[0]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)

        if teacher_forcing:
            pass
            # exp_jaw_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            # style_emb = exp_jaw_emb  
            # template = torch.zeros([1, 1, exp_jaw.shape[2]]).to(device=self.device)
            # exp_jaw_input = torch.cat((template, exp_jaw[:,:-1]), 1) # shift one position
            # exp_jaw_input = self.exp_jaw_map(exp_jaw_input)
            # exp_jaw_input = exp_jaw_input + style_emb
            # exp_jaw_input = self.PPE(exp_jaw_input)
            # tgt_mask = self.biased_mask[:, :exp_jaw_input.shape[1], :exp_jaw_input.shape[1]].clone().detach().to(device=self.device)
            # memory_mask = enc_dec_mask(self.device, exp_jaw_input.shape[1], hidden_states.shape[1])
            # exp_jaw_out = self.transformer_decoder(exp_jaw_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            # exp_jaw_out = self.exp_jaw_map_r(exp_jaw_out)
        else:
            shape_params = torch.zeros([1, 300]).to(torch.device(vertice.device))
            
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                exp_jaw_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                exp_jaw_out = self.exp_jaw_map_r(exp_jaw_out)

                pose_params = torch.zeros([batch_size, 15]).to(torch.device(vertice.device))
                pose_params[:, 6:9] = exp_jaw_out[:, -1, 50:]

                verts, _ = self.flame(batch_size, \
                                             shape_params=shape_params, \
                                             expression_params=exp_jaw_out[:, -1, :50], \
                                             pose_params=pose_params)
                
                if i==0:
                    vertice_out2 = verts.reshape(batch_size, 1, -1)
                else:
                    vertice_out2 = torch.cat((vertice_out2, verts.reshape(batch_size, 1, -1)), 1)

                new_output = self.exp_jaw_map(exp_jaw_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        criterion = torch.nn.MSELoss(reduction='none')
        loss = criterion(vertice_out2, vertice) # (batch, seq_len, V*3)
        loss *= keep_mask.unsqueeze(0)
        vertice_out = vertice_out2 - template
        loss2 = criterion(vertice_out, torch.zeros_like(vertice_out).to(device='cuda'))
        loss += loss2 * 0.01
        loss = torch.mean(loss)
        return loss

    def predict(self, audio, template, one_hot, fps):
        batch_size = audio.shape[0]
        shape_params = torch.zeros([1, 300]).to(torch.device(audio.device))
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)

        hidden_states = None
        frame_num = int(audio.shape[1]/16000*fps)
        remain_audio = audio
        to_cur_frame = 0
        audio_max_len = int(self.max_seq_len*520)
        while(remain_audio.shape[1]>0):
            if(remain_audio.shape[1]>audio_max_len):
                audio_clip = remain_audio[:, :audio_max_len]
            else:
                audio_clip = remain_audio
            remain_audio = remain_audio[:, audio_max_len:]

            cur_frame_num = round((1 - remain_audio.shape[1] / audio.shape[1]) * frame_num) - to_cur_frame
            to_cur_frame += cur_frame_num

            hidden_states_clip = self.audio_encoder(audio_clip, self.dataset, frame_num=cur_frame_num).last_hidden_state
            hidden_states_clip = self.audio_feature_map(hidden_states_clip)

            if(hidden_states is None):
                hidden_states = hidden_states_clip
            else:
                hidden_states = torch.concat([hidden_states, hidden_states_clip], dim=1)

        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                if(i%self.max_seq_len==0):
                    vertice_input = self.PPE(vertice_emb[:,-1,:].unsqueeze(1))
                else:
                    vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            
            if(i%self.max_seq_len==0):
                slice_hidden_states = hidden_states[:, i:i+self.max_seq_len, :]

            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], slice_hidden_states.shape[1])
            exp_jaw_out = self.transformer_decoder(vertice_input, slice_hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            exp_jaw_out = self.exp_jaw_map_r(exp_jaw_out)

            pose_params = torch.zeros([batch_size, 15]).to(torch.device(audio.device))
            pose_params[:, 6:9] = exp_jaw_out[:, -1, 50:]

            verts, _ = self.flame(batch_size, \
                                            shape_params=shape_params, \
                                            expression_params=exp_jaw_out[:, -1, :50], \
                                            pose_params=pose_params)
            
            new_output = self.exp_jaw_map(exp_jaw_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb

            if i==0:
                vertice_emb = torch.cat((vertice_emb, new_output), 1)
                vertice_out2 = verts.reshape(batch_size, 1, -1)
                exp_jaw_out2 = exp_jaw_out.clone().detach()
            else:
                if(i%self.max_seq_len==0):
                    vertice_emb = torch.cat((vertice_emb[:,-1,:].unsqueeze(1), new_output), 1)
                    exp_jaw_out = exp_jaw_out[:, -1, :].unsqueeze(1)
                else:
                    vertice_emb = torch.cat((vertice_emb, new_output), 1)
                vertice_out2 = torch.cat((vertice_out2, verts.reshape(batch_size, 1, -1)), 1)
                exp_jaw_out2 = torch.cat((exp_jaw_out2, exp_jaw_out[:, -1, :].unsqueeze(1)), 1)

        return vertice_out2, exp_jaw_out2
