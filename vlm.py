import torch
from torch import nn

from vision_embeddings import VisionEmbeddings
from transformer import Encoder
from text_embeddings import TextTockenEmbedding, TokenEmbedding, Tokeniser_ASCI

import numpy as np
from typing import Optional

'''
My take on VLM.
While beying similar to the PaliGemma, 
my implementation differes in some points as resing my previous Transformer and  VisionEmbeddings classes

References:
PaliGemma: A versatile 3B VLM for transfer https://arxiv.org/pdf/2407.07726
UJ video https://www.youtube.com/watch?v=vAmKB7iPkWw
'''

class VisionProjection(nn.Module):
    # class project vision token emebddings to dims of text embeddings for proper fusion
    
    def __init__(self, d_vision: int, d_text: int):
        super().__init__()
        self.d_vision = d_vision
        self.d_text = d_text

        self.proj_vis2text = nn.Linear(in_features=d_vision, out_features=d_text)

    def forward(self, x: torch.tensor)->torch.tensor:
        # x is vision emebddings, here converted to text embedings dims
        # x [B, n_vision_tokens, d_vision] 
        # y [B, n_vision_tokens, d_text] 
        y = self.proj_vis2text(x)
        return y
    
class VisionModel(nn.Module):
    # Vision Embedings->Vision Encoder->text shape projection
    # we could use VisionEncoder with properly set paramts too but this is more straightforward
    def __init__(self, vis_emb: VisionEmbeddings, enc: Encoder, proj: VisionProjection):
        super().__init__()
        assert vis_emb.is_add_CLS == False, 'CLS must be disbaled in VisionEmbeddings'
        self.vis_emb = vis_emb
        self.enc = enc
        self.proj = proj

    def forward(self, img:torch.tensor)->torch.tensor:
        img_tockens = self.vis_emb(img)
        img_emb = self.enc(img_tockens)
        img_text_shape_emb = self.proj(img_emb)

        return img_text_shape_emb

class VLM(nn.Module):
    """
    image + prompt -> answer text ouput
    VLM is copmosed of 
     - vision_tower
     - text_tower
     - Text and image emeddnigs patching
     - Decoder (actually encoder as no cross attention)
    """
    def __init__(self, 
            vision_tower: VisionModel, text_tower:TextTockenEmbedding, fusion_decoder: Encoder, 
            lm_head: Optional[nn.Module]=None):
        super().__init__()
        self.vision_tower = vision_tower
        self.text_tower = text_tower
        self.fusion_decoder = fusion_decoder

        if lm_head is None:
            d_text = text_tower.emb_enc.d_model
            vocab_size = text_tower.emb_enc.vocab_size
            lm_head = nn.Linear(in_features=d_text, out_features=vocab_size)
        self.lm_head = lm_head

    def forward(
            self, prompt: torch.tensor, n_p:int, 
            is_ans_only:bool=True, img: Optional[torch.tensor]=None, pre_vision_emb: Optional[torch.tensor]=None
                )->torch.tensor:
        # img is batch of images [B, C, H, W]
        if pre_vision_emb is None:
            assert img is not None, f"Both 'img' and 'pre_vision_emb' can't be None"
            vision_emb = self.vision_tower(img) # [B, n_vision, d_text]
        else:
            vision_emb = pre_vision_emb

        # prompt is batch of prompts tockens [B, n_text]
        #  '/n' + BOS + Prompt + EOS + answer + padding
        text_emb = self.text_tower(prompt) # [B, n_text_tockens, d_text]

        # what if batch images resoluton not the same? 
        # That's a TODO item. Either resize as paper does, or use max_len pading for shorter, croping/sampling for longer
        n_b_vision, n_vision, n_d_proj_vision = vision_emb.shape 

        # we expect prompt per batch, aligned with image. Promt lenght is fixed first. 
        # TODO variable prompt length- to be adressed by caller, making sure fixed no of tockens
        n_b_text, n_text, n_d_text = text_emb.shape

        assert  n_b_vision == n_b_text, 'Text and vision batch size misalingment.'
        assert  n_d_proj_vision == n_d_text, 'Text and vision embeddings size misalingment.'

        fused_emb = torch.cat(tensors=[vision_emb, text_emb], dim=-2) # [B, n_vision+n_text, n_d_text]
        # vision emb + BOS + promt emb + /n + ans + EOS + PAD
        n_fused = fused_emb.shape[-2]
        # n_p is the no of prompt tockens excluding EOS, BOS, can be calculated via len(enc.raw_encode_line(text))
        # The start of the answer after the '/n' 
        i_p = n_vision + 1 + n_p + 1  # vision + BOS + prompt + \n

        # create and provide casuality matrix using 
        #   n = n_vision + n_text(1 (BOS) +n_p(pompt tockens) + 1(\n) + #answer(n_pad) + 1(EOS)  ) 
        #   i_p = iloc(\n)
        # Verified with attention.py->Attention: Expected mask of bools with True indicating values to be -Inf
        device = fused_emb.device
        casuality_mask = torch.ones(size=(n_fused, n_fused), dtype=torch.bool, device=device) 
        # set to False diagnoal i==j and beneath it i<j
        casuality_mask = torch.triu(casuality_mask, diagonal=1)
        # the inputs are not required to look only to the left, bi-derectiona attntion allowed
        casuality_mask[:i_p, :i_p,] = False
        fused_att = self.fusion_decoder(x=fused_emb, mask=casuality_mask) # [B, n_fused, d_text]

        if is_ans_only:
            # use only embeddings related to answer droping image and tockens
            # and last tocken as predicting EOS
            fused_att = fused_att[:, i_p-1:-1, :] # [B, n_fused, d_text]

        logits = self.lm_head(fused_att)
        return logits

    def generate(self, img: torch.tensor, prompt: torch.tensor, n_p: int, eos_id: int, 
                 is_ans_only:bool=True,
                 max_new_tokens: Optional[int]=None, kv_cache: Optional[torch.tensor]=None)->torch.tensor:
        # Note inefficient straighforward implmentation
        # TODO implemetn efficicient KV cahche based implmentation
        # prompt: [B, max_len] fixed-length tensor (answer region initially EOS+PAD, from encode(''))
        self.eval()
        generated = prompt.clone()
        # recalculate to avoid repeating for each tpcken
        vision_emb = self.vision_tower(img)
        B, max_len = prompt.shape
        device = vision_emb.device
        is_proc = torch.ones(size=(B, ), dtype=bool).to(device=device)

        i_p = n_p + 2
        max_new_tokens = max_new_tokens or max_len-i_p
        with torch.no_grad():
            for k in range(max_new_tokens):
                # [B, n_ans, vocab]
                logits = self.forward(
                    img=None, pre_vision_emb=vision_emb[is_proc], prompt=generated[is_proc], 
                    n_p=n_p, is_ans_only=True)
                new_tocken = logits[:, k, :].argmax(-1)
                generated[is_proc, i_p+k] = new_tocken
                is_proc[is_proc.clone()] = ~(new_tocken == eos_id)
                if not is_proc.any():
                    # all samples done
                    break

        if is_ans_only:
            generated = generated[:, i_p:]
        return generated

def test(target: str='VLM'):
    # basic smoke testing
    n_b = 11
    n_vis_toc = 4
    d_vision = 256
    d_text = 128
    input_img_shape = (256, 256, 3) # H, W, C
    enc_img_shape = (224, 224, 3) # H, W, C
    enc_patch_size = 16
    n_vision_encoders = 4

    enc_h, enc_w, enc_c = enc_img_shape
    vis_emb = VisionEmbeddings(
        d=d_vision, patch_size=enc_patch_size, img_size=enc_h,
        c=enc_c, is_add_CLS=False
        )
    vision_enc = Encoder(n=n_vision_encoders, d=d_vision)
    proj = VisionProjection(d_vision=d_vision, d_text=d_text)
    vis_tower = VisionModel(vis_emb=vis_emb, enc=vision_enc, proj=proj)

    in_img_h, in_img_w, in_img_c = input_img_shape
    n_patches = int(np.ceil(in_img_h/enc_patch_size)*np.ceil(in_img_w/enc_patch_size))
    imgs = torch.randint(low=0, high=255, size=(n_b, in_img_c, in_img_h, in_img_w))

    # is this to be done by calling function of by VisionEmbeddings forward?
    imgs = imgs.float() # imgs.type(dtype=torch.float) 
    imgs = imgs/255.0

    if target == 'VisionProjection':
        # Test Vision Projection
        vis_emd = torch.randn((n_b, n_vis_toc, d_vision))
        projector = VisionProjection(d_vision=d_vision, d_text=d_text)
        vis_text_shape_emb = projector(vis_emd)

        print(f'Vision emebddings shape: {vis_emd.shape}')
        expected_shape = (n_b, n_vis_toc, d_text)
        print(f'Converted to text emebddings shape: {vis_text_shape_emb.shape}')
        print(f'Aligned with expectations? "{vis_text_shape_emb.shape==expected_shape}"')
    elif target == 'VisionModel':
        # Test vision mode
        vis_emb = vis_tower(imgs)

        expected_shape = (n_b, n_patches, d_text)

        print(f'Input images shape: {imgs.shape}')
        print(f'Image converted to text emebddings shape: {vis_emb.shape}')
        if vis_emb.shape==expected_shape:
            print(f'Aligned with expectations!')
        else:
            print(f'Expected {expected_shape}, got {vis_emb.shape}')
    elif target == 'VLM':
        max_len = int(1e3)
        toc = Tokeniser_ASCI(max_len=max_len)
        vocab_size = toc.max_enc_id + 1
        
        text_tower = TextTockenEmbedding(vocab_size=vocab_size, max_len=max_len, d=d_text)

        n_fused_decoder = 4
        fusion_decoder = Encoder(n=n_fused_decoder, d=d_text)
        vlm_model = VLM(vision_tower=vis_tower, text_tower=text_tower, fusion_decoder=fusion_decoder)

        is_ans_only = True

        prompt = 'What is the image dominant color?'
        
        prompt_raw_emb = toc.raw_encode_line(prompt)
        n_p = len(prompt_raw_emb)
        
        full_prompt = prompt + '\n'
        prompt_tockens = torch.tensor(toc.encode(full_prompt))
        # expand to align with batch images
        prompt_tockens = prompt_tockens.unsqueeze(0).expand(n_b, *prompt_tockens.shape) # repeat(n_b, 1, 1)
        vlm_responce_tockens = vlm_model.forward(img=imgs, prompt=prompt_tockens, n_p=n_p, is_ans_only=is_ans_only)

        print(f'Input images shape: {imgs.shape}')
        print(f'Input prompt emebddings shape: {prompt_tockens.shape}')
        if is_ans_only:
            expected_shape = (n_b, max_len-n_p-2, vocab_size) # drop BOS prompt  '\n'
        else:
            expected_shape = (n_b, n_patches + max_len, vocab_size)
        if vlm_responce_tockens.shape==expected_shape:
            print(f'vlm_responce_tockens {vlm_responce_tockens.shape} aligned with expectations!')
        else:
            print(f'Expected {expected_shape}, got {vlm_responce_tockens.shape}')
if __name__ == '__main__':
    test('VLM')