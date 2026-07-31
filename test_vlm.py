import torch
from torch import optim
from torch import nn

import matplotlib.pyplot as plt 
from typing import Tuple, List, Optional
import math
#  import random

from vlm import VLM, VisionEmbeddings, VisionModel, VisionProjection
from transformer import Encoder
from text_embeddings import TextTockenEmbedding, Tokeniser_ASCI

# define colors banks as dict
colors_d = {
    # color name, color RGB
    'White': [255, 255, 255],
    'Black': [0, 0, 0],
    'Red':   [255, 0, 0],
    'Green': [0, 255, 0],
    'Blue':  [0, 0, 255],
    'Yellow':  [255, 255, 0],
    'Orange':  [255, 165, 0],
    'Purple':  [128, 0, 128],
    'Cyan':    [0, 255, 255],
    'Magenta': [255, 0, 255],
    'Pink':    [255, 192, 203],
    'Brown':   [165, 42, 42],
    'Gray':    [128, 128, 128],
}

def generate_colors(colors_d: dict, n_samples: int, is_random: bool = False)->Tuple[List[str], List[List[int]]]:
    # return a list of color values RGB and color names
    color_names, color_vals = [], []
    for clr_name, clr_val in colors_d.items():
        color_names.append(clr_name)
        color_vals.append(clr_val)
    n_colors = len(colors_d)
    if is_random:
        # use torch random rather then pyhtoin random to allow repeatabl radomisaiton via torch.manual_seed(int)
        i_colors = torch.randint(low=0, high=n_colors, size=(n_samples, ))
    else:
        i_colors = torch.arange(n_samples) % n_colors
    sample_names = [color_names[i_color] for i_color in i_colors]
    sample_values = [color_vals[i_color] for i_color in i_colors]

    return sample_names, sample_values

def generate_color_images(colors: List[List[int]], img_shape: Tuple[int, int, int], clr_var:float=0, clr_scale:int=255)->torch.tensor:
    # return RGB images per each provided color and image size, with user deinfed variation
    # colors list of [[R, G, B]] 
    # [B, H, W, C] Open CV notation
    n_images = len(colors)
    size = tuple([n_images] + list(img_shape[:])) # is there a more legant way to do this?
    images = torch.zeros(size=size, dtype=torch.int)
    for i_image in range(n_images):
        images[i_image, :, :, :] = torch.tensor(colors[i_image])

    if clr_var:
        # add color variation- here per pixel, can be done uniform per whole image
        noise = clr_var*torch.randint(low=-clr_scale, high=clr_scale, size=img_shape, dtype=torch.float)
        images = images.float() + noise
        images = images.clamp(min=0, max=clr_scale).int()
    return images

def show_images(images:torch.tensor, titles:List[str], n_max_single_row_cols:int = 6, is_torch_notation: bool=True):
    # Show gird of images and their titles
    if is_torch_notation:
        # permute torch to Color last notation
        # [B, C, H, W]->[B, H, W, C]
        images = images.permute(0, 2, 3, 1)
    n_images = len(images)
    
    if n_images > n_max_single_row_cols:
        # Multi row axes
        n_cols = int(math.ceil(math.sqrt(n_images)))
        n_rows = (n_images) // n_cols
        if n_images % n_cols:
            n_rows += 1
    else:
        # single row axes
        n_rows, n_cols = 1, n_max_single_row_cols

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows)
    axes = axes.flatten()
    for i, (img, ttl) in enumerate(zip(images, titles)):
        cmap = 'gray' if img.shape[2] == 1 else None
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(ttl)
        axes[i].axis('off')

    for i_hide in range(i+1, n_images):
        # that does not seem to work
        axes[i_hide].axis('off')

    plt.tight_layout()
    plt.show()
    return

def trunc_at_eos(tockens: List[int], eos: int):
    if eos in tockens:
        return tockens[:tockens.index(eos)]
    return tockens

def calc_acc(pred_logits: torch.tensor, labels: torch.tensor, toc:Tokeniser_ASCI):
    pred_id = pred_logits.argmax(-2)
    pred_text = [toc.decode(trunc_at_eos(line, eos=toc.EOS)) for line in pred_id.tolist()]
    label_text = [toc.decode(line) for line in labels.tolist()]
    acc = 0
    for pred, label in zip(pred_text, label_text):
        acc += pred == label
    n_samples = len(label_text)
    return acc/n_samples

def train_VLM(
        model: VLM, images: torch.tensor, toc:Tokeniser_ASCI, prompt_q:str, prompt_a:List[str],
        n_epoch:int, n_batch:int = 3, lr: float=1e-3, device: torch.device=torch.device('cpu'),
        stop_loss: Optional[float]=None, stop_acc:Optional[float]=None, )->VLM:
    # TODO move data and cmopute to GPU if exist, or have it added to class?
    [n_samples, C, H, W] = images.shape
    
    # Train the model
    loss_fn = nn.CrossEntropyLoss(ignore_index=toc.PAD)
    opt = optim.Adam(params=model.parameters(), lr=lr)
    # ceiling div in plain Python -- torch.ceil needs a Tensor, not a bare int
    n_batches = -(-n_samples // n_batch) # int(torch.ceil(torch.tensor(n_samples) / n_batch)) #   
    prompts = [f'{prompt_q}\n{prompt_a[i]}' for i in range(n_samples)]
    
    n_p = len(toc.raw_encode_line(prompt_q)) # fixed constant -- prompt never varies

    prompts_toc = torch.tensor([toc.encode(prompt) for prompt in prompts])
    images = images.to(device)
    prompts_toc = prompts_toc.to(device)

    max_len = prompts_toc.shape[-1]
    for i_epoch in range(n_epoch):     
        # modify labels if train data is shuffled   
        idx = torch.randperm(n_samples)
        images_batch = images[idx].reshape(n_batches, n_batch, C, H, W)
        prompts_toc_batch = prompts_toc[idx].reshape(n_batches, n_batch, max_len)
        labels = prompts_toc_batch[:, :, n_p+2:] # (text-local answer start = 1(BOS)+n_p+1('\n')), sliced per-batch

        epoch_loss = 0
        epoch_acc = 0
        for i_batch in range(n_batches):
            # loop over batches
            model.train()
            
            # model should send tensors to device?
            logits = model.forward(
                img=images_batch[i_batch], 
                prompt=prompts_toc_batch[i_batch], 
                n_p=n_p, 
                is_ans_only=True)
            # make pres aligned with labels
            # [B, n_fused/n_ans, vocab_size] -> [B, vocab_size, n_fused/n_ans]
            preds = logits.permute(0, 2, 1)
            
            batch_loss = loss_fn(preds, labels[i_batch])

            epoch_loss += batch_loss.item()
            # max across vocab dimention
            epoch_acc += calc_acc(pred_logits=preds, labels=labels[i_batch], toc=toc)
            # [toc.decode(line) for line in labels[i_batch].tolist()]
            model.zero_grad()
            batch_loss.backward()
            opt.step()

        # report loss and acc during trainig
        u_loss = epoch_loss/n_batches
        u_acc = epoch_acc/n_batches
        print(f'Epoch {i_epoch+1}/{n_epoch} loss: {u_loss:.3f}, acc: {u_acc:.3f}')
        if stop_loss is not None and u_loss < stop_loss:
            print(f'Reached target loss < {stop_loss:.3f}, stopping')
            break
        if stop_acc is not None and u_acc > stop_acc:
            print(f'Reached target acc > {stop_acc:.3f}, stopping')
            break
    # retrn trained model
    return model

def pred_eval(model: VLM, images: torch.tensor, toc:Tokeniser_ASCI,
              prompt_q:str, prompt_a:List[str], 
              actual_labels:Optional[List[str]]=None, n_eval_samples: Optional[int]=None,
              is_generate: bool = False, device: torch.device=torch.device('cpu')):
    # infer with trained model and show predicted answers
    n_samples = len(images)
    model.eval()

    prompts = [f'{prompt_q}\n{prompt_a[i]}' for i in range(n_samples)]
    n_p = len(toc.raw_encode_line(prompt_q)) # fixed constant -- prompt never varies
    prompts_toc = torch.tensor([toc.encode(prompt) for prompt in prompts])

    if not n_eval_samples or n_eval_samples > n_samples or n_eval_samples < 0: 
        n_eval_samples = n_samples # n_samples//3
    i_eval = torch.arange(n_samples)[:n_eval_samples]
    images = images.to(device)
    prompts_toc = prompts_toc.to(device)
    if is_generate:
        # geneated tockens in this case
        preds_tockens = model.generate(
            img=images[i_eval], 
            prompt=prompts_toc[i_eval], 
            n_p=n_p,
            eos_id=toc.EOS,
            is_ans_only=True)
        preds_tockens = preds_tockens.tolist()
    else:
        logits = model.forward(
            img=images[i_eval], 
            prompt=prompts_toc[i_eval], 
            n_p=n_p, 
            is_ans_only=True)
        
        # make pres aligned with labels
        # logits [B, n_fused/n_ans, vocab_size]
        preds_tockens = logits.argmax(-1).tolist()
    p_anses = prompts_toc[i_eval, n_p+2:].tolist()
    eval_titles = [f"Pred-'{toc.decode(trunc_at_eos(pred, eos=toc.EOS))}', Provided label- '{toc.decode(p_ans)}'" for pred, p_ans in zip(preds_tockens, p_anses)]
    if actual_labels:
        assert len(eval_titles) == len(actual_labels)
        eval_titles = [f"{ttl}, Actual label '{lbl}'" for ttl, lbl in zip(eval_titles, actual_labels)]
    print('\n'.join(eval_titles))
    #  show_images(images=images, titles=eval_titles, n_max_single_row_cols=3, is_torch_notation=True)
    
    # retrn trained model
    return model


def img_color_test(device: torch.device, n_samples = 6,):
    # create batch of images and aligned prompts
    # generate toy dataste of colored images
    torch.manual_seed(12)
    n_batch = 3
    n_samples = (n_samples//n_batch)*n_batch # make sure we can split the inputs to bathes cleanly
   
    img_shape = [64, 64, 3]
    clr_scale = 255
    sample_names, sample_values = generate_colors(colors_d, n_samples=n_samples, is_random=False)
    images = generate_color_images(colors=sample_values, img_shape=img_shape, clr_var=0, clr_scale=clr_scale)
    
    # convert to torch notation [B, C, H, W]
    images = images.permute(0, 3, 1, 2)
    [B, C, H, W] = images.shape

    # print(f'Images shape {images.shape}')
    # print(f'Image shape {images[0].shape}')
    # for i_sample in range(n_samples):
    #     print(f'Image color {sample_names[i_sample]}')
    #     # print(f'\n{images[i_sample]}\n')        
    #     print(f'\n{images[i_sample, :, :, :]}\n')
    # show_images(images=images, titles=sample_names, is_torch_notation=True)

    # norm image values to [0.0, 1.0]
    images = images.float()/clr_scale
    # create prompts and answers
    prompt_q = 'What is the image color?'

    # build VLM compenets
    # configs
    # vision
    d_vision = 64
    n_vision = 4

    # text
    max_len = 130
    toc = Tokeniser_ASCI(max_len=max_len)
    d_text = 128
    vocab_size = toc.max_enc_id + 1

    # fusion
    n_fusion = 4

    # we expct train images to be of same dims, and have W=H
    img_size=max(H, W) # for not rectangular inpus use max dim to avoid extra token patches
    patch_size = max(8, min(img_size//8, 16)) # patch_size [8, 16]. 
    vis_emb = VisionEmbeddings(d=d_vision, is_add_CLS=False, c=C, img_size=img_size, patch_size=patch_size)
    vis_enc = Encoder(n=n_vision, d=d_vision)
    vis_proj = VisionProjection(d_vision=d_vision, d_text=d_text)
    vision_tower=VisionModel(vis_emb=vis_emb, enc=vis_enc, proj=vis_proj)

    text_tower = TextTockenEmbedding(vocab_size=vocab_size, max_len=max_len, d=d_text)

    fusion_decoder = Encoder(n=n_fusion, d=d_text)
    model = VLM(vision_tower=vision_tower, text_tower=text_tower, fusion_decoder=fusion_decoder)

    print(f'Running "img_color_test" on {device.type}')
    model.to(device)

    train_VLM(
        model=model, images=images, prompt_q=prompt_q, prompt_a=sample_names, toc=toc, 
        n_batch=n_batch, n_epoch=100, lr=1e-3, stop_loss=5e-3, stop_acc=0.95, device=device)

    print('\nOver-fitting test')
    pred_eval(model=model, images=images, prompt_q=prompt_q, prompt_a=sample_names, toc=toc,
              is_generate=True, device=device)

    # ablation test
    print('\n\nAblation test')
    img_noise = torch.randint(low=0, high=255, size=(1, H, W, C))
    img_pink = generate_color_images(colors=[[255, 192, 203]], img_shape=(H, W, C), clr_var=0.0, clr_scale=clr_scale)
    img_noisy_gray = generate_color_images(colors=[[128, 128, 128]], img_shape=(H, W, C), clr_var=0.2, clr_scale=clr_scale)
    img_RGB = generate_color_images(
        colors=[[255, 0, 0],
                [0, 255, 0],
                [0, 0, 255]], 
        img_shape=(H, W, C), clr_var=0.0, clr_scale=clr_scale)

    abl_images = torch.cat([img_noise, img_pink, img_noisy_gray, img_RGB], dim=0)
    abl_images = abl_images.permute(0, 3, 1, 2).float()/clr_scale
    n_samples = len(abl_images)
    actual_clrs = ['Noise', 'Pink', 'Noisy Gray', 'Red', 'Green', 'Blue']
    color_a = '' # 'White' ''
    prompt_a = [color_a]*n_samples 
    pred_eval(model=model, images=abl_images, prompt_q=prompt_q, prompt_a=prompt_a, toc=toc, actual_labels=actual_clrs, 
              is_generate=True, device=device)
    return

if __name__ == '__main__':
    # Converges for 6, but not for 12
    n_samples=6     # len(colors_d)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    img_color_test(n_samples=n_samples, device=device)