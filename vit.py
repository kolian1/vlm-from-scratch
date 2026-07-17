import torch
from torch import nn
import torch.nn.functional as F

from vision_embeddings import VisionEmbeddings
from transformer import Encoder

'''
Goal, impelement ViT
References 
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy et al. https://arxiv.org/abs/2010.11929
'''
'''
Impelmentation
input mebeddings->encoding layers->MLP->class precition

'''
class VisionEncoder(nn.Module):
    def __init__(self, vision_emb: VisionEmbeddings, att_encoder: Encoder, mlp_head: nn.Module=None, n_class: int=None):
        super().__init__()
        self.vision_emb = vision_emb
        self.att_encoder = att_encoder
        if not mlp_head and n_class:
            # default mlp_head, single layer
            mlp_head = nn.Linear(in_features=vision_emb.d, out_features=n_class)
        self.mlp_head = mlp_head
  
    def forward(self, img: torch.tensor)->torch.tensor:
        emb_img = self.vision_emb(img)
        att_emb_img = self.att_encoder(emb_img)
        cls_emd = att_emb_img[:, 0,:]
        pred = self.mlp_head(cls_emd)
        return pred
 

def test():
    B = 5
    C = 3
    H = 240
    W = 300

    sim_img = torch.randn(size=(B, C, H, W))

    d = 512
    n_enc_layers = 6
    n_class = 10
    # Smoke test
    vision_emb = VisionEmbeddings(d=d)
    att_encoder = Encoder(d=d, n=n_enc_layers)
    mlp_head = nn.Sequential(
        nn.Linear(in_features=d, out_features=64), 
        nn.ReLU(), 
        nn.Linear(in_features=64, out_features=n_class)
        )
    vit = VisionEncoder(vision_emb=vision_emb, att_encoder=att_encoder, mlp_head=mlp_head)
    v_emb = vision_emb(sim_img)
    pred = vit(sim_img)
    print(f'Input dims {sim_img.shape}')
    print(f'Visual embeddings dims {v_emb.shape}')
    print(f'Predictions  dims {pred.shape}')

if __name__ == '__main__':
    test()