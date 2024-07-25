import math
from numpy import size
import torch
from torch import nn
import torch.nn.functional as F

def attention(query, value, key_ = None):
    """ Info. Ref. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
        "If key not given, will use value for both key and value, which is the most common case."
        
        query, key, value: (n_batch, seq_len, d_k)
        mask: (n_batch, seq_len, seq_len)
    """
    
    if key_ is None:
        key = value
    else:
        key = key_
    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
    return out

########## Feature Extraction
class sub_fe(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 20,
                              kernel_size = 10, stride=1)
        self.conv2 = nn.Conv1d(in_channels = 20, out_channels = 40,
                              kernel_size = 5, stride=1)
        self.conv3 = nn.Conv1d(in_channels = 40, out_channels = 80,
                              kernel_size = 3, stride=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.15)

        self.lstm1 = nn.LSTM(80, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dense = nn.Linear(in_features=32, out_features=4)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = torch.permute(x, (0, 2, 1)) 
        
        x, _ = self.lstm1(x) # Batch, sequence, input_size(channel)
        x, _ = self.lstm2(x)
        x = x[:, -1, :] # return_sequences=False -> last sequence
        
        x = self.dense(x)
        return x

########## Attention
class sub_attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attention = attention
        self.dense = nn.Linear(hidden, 4)

    def forward(self, input): # input: a tensor or list of tensor 
        if type(input) == list:
            h = torch.cat(input, dim=1) # unimodal feature vector가 여러개면 하나로 concatenate
        else:
            h = torch.unsqueeze(input, dim=2) # unimodal feature vector가 한개면 그냥 그대로 사용
        
        # Query-Value Attention sequence
        qv_attention_seq = self.attention(h, h)
        
        s = torch.cat([h, qv_attention_seq], dim=1)
        s = torch.squeeze(s)
        
        output = self.dense(s)
        
        return output



########## Tensor Fusion
class tfn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(0.15)
        self.dense1 = nn.Linear(125, 32)
        self.dense2 = nn.Linear(32, self.args.n_classes)
        
        
    def forward(self, inputs):
        x1, x2, x3 = inputs
        
        bias = torch.ones_like(x1)[:, -1:] # inputs: [torch(4), torch(4), torch(4)] -> 피험자3-실험2의 valid dataset

        biased_x1 = torch.cat([x1, bias], dim=1)
        biased_x2 = torch.cat([x2, bias], dim=1)
        biased_x3 = torch.cat([x3, bias], dim=1)
        
        unsqueezed_x1 = torch.unsqueeze(biased_x1, dim=1)
        unsqueezed_x2 = torch.unsqueeze(biased_x2, dim=2)
        unsqueezed_x3 = torch.unsqueeze(biased_x3, dim=2)
        
        x1_by_x2 = torch.matmul(unsqueezed_x2, unsqueezed_x1)
        reshaped_x1x2 = torch.reshape(x1_by_x2, (x1_by_x2.shape[0], 1, -1))
        
        fused = torch.matmul(unsqueezed_x3, reshaped_x1x2)
        fusion = torch.reshape(fused, (fused.shape[0], -1))
        
        fusion = self.dropout(fusion)
        fusion = self.dense1(fusion)
        fusion = self.dropout(fusion)
        fusion = self.dense2(fusion)
        output = F.softmax(fusion, dim=1)

        return output


########## Proposed model
class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sub_fe1 = sub_fe(32)
        self.sub_fe2 = sub_fe(1)
        self.sub_fe3 = sub_fe(1)

        self.sub_attn1 = sub_attn(8)
        self.sub_attn2 = sub_attn(8)
        self.sub_attn3 = sub_attn(8)
        self.tfn = tfn(args)
        
    def forward(self, inputs):
        (xe, xp, xc) = inputs
        (xe_fe, xp_fe, xc_fe) = self.sub_fe1(xe), \
                                self.sub_fe2(xp.unsqueeze(dim=1)), \
                                self.sub_fe3(xc.unsqueeze(dim=1))

        (x_attn_e, x_attn_p, x_attn_c) = self.sub_attn1(xe_fe), \
                                           self.sub_attn2(xp_fe), \
                                           self.sub_attn3(xc_fe)
        output = self.tfn([x_attn_e, x_attn_p, x_attn_c])

        return output