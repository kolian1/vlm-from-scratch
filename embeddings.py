'''
Build the encoding needed for Attention module 
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


'''
Transformer references 
Paper https://arxiv.org/abs/1706.03762
diagram 

'''
'''
07/07/26
Moiving on towards the transfomer- implementing token and posiitona encoding
The high level
text input-> tokenizer->d vector embeddings->+ positinal encoding->attention/transformed input
'''
class TokenEmbedding(nn.Module): # (nn.Embedding)
    '''
    Convert integer inputs to embeddings
    (B, T) ints → (B, T, d_model)
    '''
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__() # is it needed?
        self.d_model = d_model
        self.vocab_size = vocab_size

        # for an input integer, return ouput vector[d_model]
        # possible to implement via Linearlayer with trainable weights 
        # self.linearEnc = nn.Linear(in_features=1, out_features=d_model)
        # alternative, hypotetical has table of seen int as key, and value [d_model], but how to train it? 
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model) # an embedding stable [vocab_size, d_model]
        
    def forward(self, x:torch.tensor)->torch.tensor:
        # in (B, T) ints → out (B, T, d_model)
        x_emb = self.emb(x)
        return x_emb

class PositionalEncoding(nn.Module): # (nn.Embedding)
    '''
    Add positinal encoding for each word,
    (B, T, d_model) → (B, T, d_model)
    '''

    def __init__(self, max_len:int, d_model: int):
        super().__init__() # is it needed?
        # pre-compute the
        self.max_len = max_len
        self.d_model = d_model # d in equations
        # PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
        # PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
        
        i_pos = torch.arange(0, d_model, 2) # 2i: 0, 2, 4, ... 2*(d//2)
        # 1/(1e4^i/d) = 1e4^((-i)/d) = exp(-i*log(1e4/d))
        freq_mul = torch.exp(-(i_pos/self.d_model)*math.log(1e4)) # [d//2]
        # .unsqueeze(0) converts [d//2] to [1, d//2]
        freq_mul = freq_mul.unsqueeze(0) # [1, d//2]
        
        pos = torch.arange(0, self.max_len).type(torch.float) # [max_len]
        # .unsqueeze(1) converts [max_len] to [max_len, 1]
        pos = pos.unsqueeze(dim=1) # []
        # for each position provide d_model pos embeddings
        angles = pos@freq_mul # [max_len, 1]@[1, d//2] = [max_len, d//2]
        # angles = pos*freq_mul # [max_len, 1]*[1, d//2] = [max_len, d//2]
        pos_enc = torch.zeros(size=(self.max_len, self.d_model))
        self.register_buffer('pos_enc', pos_enc)

        # even d//2 elemets along dim=1 emdedding dimention
        self.pos_enc[:, i_pos] = torch.sin(angles) 
        # odd d//2 elemets along dim=1 emdedding dimention
        self.pos_enc[:, i_pos+1] = torch.cos(angles)
        return
    
    def forward(self, x:torch.tensor)->torch.tensor:
        n_x = x.shape[-2]
        assert n_x <= self.max_len, f'Input lenght {n_x} excedded maximla supported lenght {self.max_len}'
        i_pos = torch.arange(0, n_x)
        shift_pos_enc = self.pos_enc[i_pos]
        y = x + shift_pos_enc
        return y

def testing(): 
    # testing
    n_b = 2 # no of batches
    n_t = 5 # no of tocken in  batch
    d_model = 16
    vocab_size = 500
    max_len=int(1e3)

    # random integres representing tockenizer vocab tokens
    x= torch.randint(0, vocab_size, size=(n_b, n_t))

    toc2emb = TokenEmbedding(d_model=d_model, vocab_size=vocab_size)
    x_emb = toc2emb.forward(x)
    print(f'Token dims {x.shape} d_model={d_model}, out emd dims {x_emb.shape}')

    pos_emb = PositionalEncoding(max_len=max_len, d_model=d_model)

    # some pos encoder tests
    min_vals, max_vals = pos_emb.pos_enc.min(axis=1)[0], pos_emb.pos_enc.max(axis=1)[0]

    print('Pos endoder values should be [-1, 1]')
    print(f'min:{min_vals.min()}\nmax:{max_vals.max()}')
    print('Pos encoing of 0 should be 0')
    for i in [0, 6]:   
        print(f'min[{i}]:{min_vals[i]}\nmax[{i}]:{max_vals[i]}')

    x_pos_emb = pos_emb.forward(x_emb)
    print(f'Token dims {x.shape} d_model={d_model}, out positnial emd dims {x_pos_emb.shape}')

if __name__ =='__main__':
    testing()