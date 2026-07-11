import torch
import torch.nn as nn # is same as previous?
from attention import MHAttention
from embeddings import TokenEmbedding, PositionalEncoding

'''
Implementing the remaing Transomfr parts
Add and norm y = att(x), z = LayerNorm(x + y)
FF Linear(d,d)
whole transofer arhitecture in the paper
Buld encoder- SABlock
Then decorer- SABlock + more 
'''

class FF(nn.Module):
    def __init__(self, d: int, d_ff: int=None, act=nn.ReLU()):
        super().__init__()

        if not d_ff:
            d_ff = int(4*d)

        # should we make actovation a parametr, or just code it?
        self.d = d
        self.d_ff = d_ff

        #  self.seq = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d)) # Is same as below?
        self.up_scale = nn.Linear(d, d_ff)
        self.act = act
        self.down_scale = nn.Linear(d_ff, d)
    
    def forward(self, x: torch.tensor)->torch.tensor:
        x = self.up_scale(x)    # [b, n, d]-> [b, n, d_ff]
        x = self.act(x)
        y = self.down_scale(x)
        return y

class EncLayer(nn.Module):
    '''
    x[b bacthes, n tokens]->x_emd[b,n,d model emdgings]->x_pos_enc->MultiheadSA(x_pos_enc)->add_norm
    '''
    def __init__(self, d:int, d_ff: int=None, h:int=4):
        super().__init__()

        self.d = d # model tocken emebdings
        self.d_ff = d_ff # FF upscale dim
        self.h = h

        self.muli_h_att = MHAttention(name='Self Attention', d=d, h=h)

        self.sa_norm = nn.LayerNorm(normalized_shape=d)
        self.ff = FF(d=d, d_ff=d_ff)

        self.ff_norm = nn.LayerNorm(normalized_shape=d)
    
    def forward(self, x:torch.tensor, mask = None)->torch.tensor:
        y = self.muli_h_att(q=x, k=x, v=x, mask=mask)   # x[b, n, d] -> y[b, n, d]

        # Add and norm afer SA
        # should i feed via self.sa_norm(x) or self.sa_norm.forward(x)
        x = self.sa_norm(x + y)     # x[b, n, d] -> y[b, n, d]

        # feed forward
        y = self.ff(x)
        
        # add and norm after FF
        y = self.ff_norm(x+y)
        
        return y

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, max_len:int, n:int, d:int, d_ff: int=None, h:int=4):
        super().__init__()

        self.emb_enc = TokenEmbedding(d_model=d, vocab_size=vocab_size)
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d)
        self.enc_list = nn.ModuleList(EncLayer(d=d, d_ff=d_ff, h=h) for _ in range(n))
    
    def forward(self, x: torch.tensor, mask: torch.tensor=None)->torch.tensor:
        x = self.emb_enc(x)
        x = self.pos_enc(x)

        for enc in self.enc_list:
            x = enc(x, mask)
        return x

class DecLayer(nn.Module):
    def __init__(self, d:int, d_ff: int=None, h:int=4):
        super().__init__()

        self.d = d # model tocken emebdings
        self.d_ff = d_ff # FF upscale dim
        self.h = h

        self.out_att = MHAttention(name='Out Attention', d=d, h=h)
        self.out_norm = nn.LayerNorm(normalized_shape=d)

        self.mixed_att = MHAttention(name='In gated Attention', d=d, h=h)
        self.mixed_norm = nn.LayerNorm(normalized_shape=d)
        self.ff = FF(d=d, d_ff=d_ff)

        self.ff_norm = nn.LayerNorm(normalized_shape=d)
    
    def forward(self, x:torch.tensor, x_enc: torch.tensor, mask_out:torch.tensor=None, mask_mixed:torch.tensor=None)->torch.tensor:
        '''
        out MHAtt-> addnorm-> input gated MHAt->addnorm->FF->addnorm
        '''
        y =  self.out_att(q=x, k=x, v=x, mask=mask_out) 
        x = self.out_norm(x+y)
        y = self.mixed_att(q=x, k=x_enc, v=x_enc, mask=mask_mixed)
        x = self.mixed_norm(x+y)
        y = self.ff(x)
        y = self.ff_norm(x+y)
        return y
    
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, max_len:int, n:int, d:int, d_ff: int=None, h:int=4):
        super().__init__()

        self.emb_enc = TokenEmbedding(d_model=d, vocab_size=vocab_size)
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d)
        self.dec_list = nn.ModuleList(DecLayer(d=d, d_ff=d_ff, h=h) for _ in range(n))
        self.dec_linear = nn.Linear(d, vocab_size)
    
    def forward(self, x:torch.tensor, mask_out:torch.tensor, x_enc: torch.tensor)->torch.tensor:
        x = self.emb_enc(x)
        x = self.pos_enc(x)
        for dec in self.dec_list:
            x = dec(x=x, mask_out=mask_out, x_enc=x_enc)
        
        x = self.dec_linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, max_len:int, n:int, d:int, d_ff: int=None, h:int=4):
        super().__init__()
        self.encoder = Encoder(vocab_size=vocab_size, max_len=max_len, n=n, d=d, d_ff=d_ff, h=h)
        self.decoder = Decoder(vocab_size=vocab_size, max_len=max_len, n=n, d=d, d_ff=d_ff, h=h)
        
        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).type(torch.bool)
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x:torch.tensor, out:torch.tensor)->torch.tensor:
        x_enc = self.encoder(x)
        # build the casual mask, bool of dims 
        n_out = out.size(-1)
        y = self.decoder(x_enc=x_enc, x=out, mask_out=self.causal_mask[:n_out, :n_out])
        return y

def test():
    # testing

    # size test
    b=3
    d=512
    d_ff = d*4
    n_in_words = 5
    n_out_words = 8
    n_layers = 6

    vocab_size=int(1e4)
    max_len = int(2e3)
    # x = torch.randn(size=(b, n_words, d))
    x = torch.randint(low=0, high=vocab_size, size=(b, n_in_words))
    y = torch.randint(low=0, high=vocab_size, size=(b, n_out_words))
    trans = Transformer(max_len=max_len, vocab_size=vocab_size, n=n_layers, d=d, d_ff=d_ff)
    y_pred = trans(x=x, out=y)

    print(f'x dims {x.shape}')
    print(f'y dims {y.shape}')
    print(f'y_pred dims {y_pred.shape}')
    
    # Causality: changing a FUTURE token must NOT change earlier outputs
    out1 = y.clone()
    out2 = y.clone()
    out2[:, -1] = (out2[:, -1] + 1) % vocab_size      # perturb only the LAST token
    y1 = trans(x=x, out=out1)
    y2 = trans(x=x, out=out2)
    print(torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5))   # must be True

if __name__ == '__main__':
    test()