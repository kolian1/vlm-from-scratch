'''
Build multi head attention using basic torch modules nn.Linear, nn.soft_max
inspired by https://www.youtube.com/watch?v=bCz4OMemCcA
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


'''
high level here before coding, a
Attention class, to store d, N layers (where do they go) ?init with inpu init 3 linear layers Q,K,V
Attention forward. 
    rececing input, ruing through Q, K, V of dims [n_seq, d] 
    resturning  soft_max(Q@KT/SQRT(d))@V- TODO- verify dims are good.
Then I would work on multi head adaptation- creating attenton per head, and applying them per head- in a loop or something. 
The if needed build the encoder self attantoin, decoder self attenton (with the heads under the hood) and cross atention. 
'''

class Attention(nn.Module):
    n_batch:int=1
    n_inps:int=0

    def __init__(self, name:str, d_in:int=512, d_out:int=512):
        super().__init__()

        self.name = name
        self.d_in = d_in
        self.d_out = d_out

        self.sqrt_d = math.sqrt(self.d_out)

        # init layers
        self.wQ = nn.Linear(self.d_in, self.d_out)
        self.wK = nn.Linear(self.d_in, self.d_out)
        self.wV = nn.Linear(self.d_in, self.d_out)
    
    def __repr__(self):
        return f"Atention layer. Name: {self.name}, d_in, d_out: ({self.d_in, self.d_out})"
    
    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask: torch.tensor=None)->torch.tensor:
        # # Q, K, V shpas are [n_batch, n_inp, d]
        # # the linear Q'=W@Q [n_inp, d_in][d_in,d_out] = [n_inp, d_out]
        q = self.wQ(q) # [n_inp, d_out]
        k = self.wK(k)
        v = self.wV(v)

        res = q@k.transpose(-1,-2) # [n_inp, d_out]@[n_inp, d_out]T=[n_inp, n_inp]
        res = res/self.sqrt_d
        if mask is not None:
            # here mask off bool,indicating what values to set to -Inf
            res = res.masked_fill(mask, float('-inf'))
            # # laternatively we can use mask of vals, with top diag -Inf
            # res = res + mask

        res = F.softmax(res, dim=-1) # last dim ignoring batch
        res = res@v # [n_inp, n_inp]@[n_inp, d]=[n_inp, d]
        return res

class MHAttention(nn.Module): # (Attention)
    def __init__(self, name:str, d:int=512, h:int=4):
        super().__init__()
        self.name = name
        self.d_in = d
        self.d_out = d//h
        self.h = h

        assert d%h == 0, f'Size mismathc, d {d} cant be split to h {h} heads, fix valuesgit '

        # head attention Linear layers
        self.att_lst = nn.ModuleList(
            Attention(name=f'head_{i_h}', d_in=self.d_in, d_out=self.d_out) 
            for i_h in range(self.h))
        # # An intutive Pyhton alternative
        # self.att_lst = [
        #     Attention(name='head_{i_h}', d_in=self.d_in, d_out=self.d_out)
        #     for i_h in range(h)]

        # layer to merge/mix the head ouputs togetehr
        self.w_out = nn.Linear(self.d_in, self.d_in)

    def __repr__(self):
        return f"MH Atention layer. Name: {self.name}, d_in, d_out: ({self.d_in, self.d_out}), h: {self.h}"

    def forward(self, q, k, v, mask = None):
        res_h = [att_model.forward(q, k, v, mask) for att_model in  self.att_lst]
        # res[i] dims [n_inp, d_out], need to cocnatenate by last dimention

        # combine head ouputs
        res = torch.cat(res_h, dim=-1)

        # mix thos heads together
        res = self.w_out(res)
        return res

# testing
name= 'trivial_att' 
d=512
n_words = 8
h=4 # number of heads
att = Attention(name='single headed attention', d_in=d, d_out=d)
mh_att = MHAttention(name='multi headed attention', d=d, h=h)
x = torch.randn(size=(n_words, d))

is_print_res = False
for mdl in [att, mh_att]:
    res = mdl.forward(q=x,k=x,v=x)
    if is_print_res:
        print(f"{mdl}\n{x}\n{res}\n")
    n_train_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f'\n\nModel {mdl}.\nModel has {n_train_params} trainable params')

