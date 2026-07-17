import torch
from torch import nn
import torch.nn.functional as F


'''
Gaol: build image emdegings egenrator
Impelmentation
image → non-overlapping patches → linear projection → sequence of patch tokens + positional encoding

'''
class VisionEmbeddings(nn.Module):
    def __init__(self, d: int, patch_size:int=16, img_size: int=256, c:int=3, verbose:int=5, rand_scale: float=0.02):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.d = d
        self.verbose = verbose
        self.rand_scale = rand_scale

        # pre build bilinear grid
        n_dim_patch = img_size // patch_size
        n_patch = n_dim_patch*n_dim_patch
        self.n_patch = n_patch
        self.n_dim_patch = n_dim_patch
        
        # Linear projection- implement linear or conv2d?
        # Linear, precedd by flatten
        in_features = patch_size*patch_size*c
        self.linear_lp = nn.Linear(in_features=in_features, out_features=d)
        
        # # conv2d followed by reshape
        # self.conv_lp = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=patch_size, stride=patch_size) # , padding=

        # pos endocing
        # self.pe = PositionalEncoding(max_len=n_patches*n_patches+1, d_model=d)
        if rand_scale:
            self.pe_table = nn.Parameter(rand_scale*torch.randn(size=(n_dim_patch, n_dim_patch, d)))
            self.CLS = nn.Parameter(rand_scale*torch.randn(size=(1, 1, d))) # randn or zeros?
        else:
            self.pe_table = nn.Parameter(torch.randn(size=(n_dim_patch, n_dim_patch, d)))
            self.CLS = nn.Parameter(torch.randn(size=(1, 1, d))) # randn or zeros?
    
    def _pad_img(self, img: torch.tensor)->torch.tensor:
        # if image dims are not an whole multiply of self.patch_size
        s_rows, s_cols = self.patch_size, self.patch_size
        H, W = img.shape[-2:]
        # chek if devision has reminder, how many pixels need to be filled.
        # note for 0 reminder we will get s_rows, s_cols 
        pad_rows = s_rows - H % s_rows
        pad_rows = pad_rows % s_rows
        pad_cols = s_cols - W % s_cols
        pad_cols = pad_cols % s_cols

        if pad_rows or pad_cols:
            # pading needed
            if self.verbose > 1:
                print(f'Padding {pad_rows} bottom rows, {pad_cols} right cols) to images')
            # padding zeros, to right & botttom
            pad = (0, pad_cols, 0, pad_rows) # left,  right, top, bottom
            img = F.pad(img, pad=pad, mode='constant', value=0)
        return img
    
    def _patch_img(self, img: torch.tensor)->torch.tensor:
        # create a tensor of  [self.n_p, self.n_p, c] 
        # Expetcing images to be of fomrat [B, C, H, W] claling method to accomomdate- unsqueeze(0), permute(2,0,1)  
        # unfold(dimension, size, step)  
        [B, C, H, W] = img.shape
        n_px = self.patch_size
        patches = img.unfold(dimension=-2, size=n_px, step=n_px) # H dim [B, C, H//n, W, n] # patch hight at dim -1
        patches = patches.unfold(dimension=-2, size=n_px, step=n_px) # W dim [B, C, H//n, W//n, n, n] # patch width at dim -1
        patches = patches.contiguous().view(B, C, -1, n_px, n_px) # patches.reshape(B, C, -1, n_px, n_px) [B, C, N_paches, n, n]
        patches = patches.transpose(-3, -4)     # patches.permute(0, 2, 1, 3, 4) [B, N_paches, C, n, n]
        return  patches
    
    def _linear_project(self, patches:torch.tensor)->torch.tensor:   
        # Run via linear projection + flatten depends on selected method
        flat_p = patches.flatten(start_dim=-3)
        enc_p = self.linear_lp(flat_p)
        return enc_p
    
    def _pos_encode(self, h:int, w:int)->torch.tensor:             
        # get the grid for the given image, per its resolution
        n_patches_w = w//self.patch_size
        n_patches_h = h//self.patch_size
        
        if n_patches_w == self.n_dim_patch and n_patches_h == self.n_dim_patch:
            # is same resolution, grab tabel as is
            return self.pe_table.reshape(self.n_patch, self.d)

        # different resolution, interpolate the table
        pe = self.pe_table.permute(2, 0, 1) # [H, W, d]-> [d, H, W]
        pe = pe.unsqueeze(0)       # [1, d, H, W]
        # [1, d, n_patches_h, n_patches_w]
        pe = F.interpolate(pe, size=(n_patches_h, n_patches_w), mode='bicubic', align_corners=False)
        # [1, d, n_patches_h, n_patches_w]->[d, n_patches_h, n_patches_w]->[n_patches_h, n_patches_w, d]
        pe = pe.squeeze(0).permute(1, 2, 0)     
        
        return pe.reshape(-1, self.d) # pe.reshape(n_patches_h*n_patches_w, self.d) 


    def forward(self, img: torch.tensor)->torch.tensor:
        # not adding the
        # run above methods
        # posisble make the private _patch_img
        # use tmp\embeddings.py methods to generte positn encoding for n_patches
        # sum and return
        img = self._pad_img(img)
        (B, C, H, W) = img.shape # img [B, C, H, W]
        x = self._patch_img(img=img)
        x = self._linear_project(patches=x)
        # pos endoce all patches, skipping CLS
        x = x + self._pos_encode(h=H, w=W)
        # add CLS
        exp_LLS = self.CLS.expand(B, -1, -1)
        y = torch.cat([exp_LLS, x], dim=-2) # [B, n+1, d]

        return y

def test():
    B = 5
    C = 3
    H = 240
    W = 300

    sim_img = torch.randn(size=(B, C, H, W))

    d = 512
    # Smoke test
    patch_emd = VisionEmbeddings(d=d)
    img_enc = patch_emd(sim_img) # .forward(patch_emd)

    print(f'Input dims {sim_img.shape}')
    print(f'Emeddeed pathes dims {img_enc.shape}')

if __name__ == '__main__':
    test()