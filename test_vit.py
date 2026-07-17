import torch
from torch import nn
import torchvision

import matplotlib.pyplot as plt
import math

from vit import VisionEncoder
from vision_embeddings import VisionEmbeddings
from transformer import Encoder

def inspect_dataset(dataset: torch.utils.data.Dataset):
    print(f'\n{"#"*10} Inspect Dataset {"#"*10}\n')
    # print(dir(dataset))

    print(f'No of images in dataset {len(dataset)}')
    print(f'Dataset classes {dataset.classes}')
    print(f'Dataset classes count: {torch.bincount(dataset.targets)}')
    print(f'dataset.data.shape: {dataset.data.shape}')
    img, label = dataset[0]
    print(f'\nImage shape {img.shape}, label shape {label}')
   

def inspect_dataloader(dataloader: torch.utils.data.DataLoader):
    print(f'\n{"#"*10} Inspect DataLoader {"#"*10}\n')
    print(f'Datalodrr length- {len(dataloader)}')                   # number of BATCHES, not images
    images, labels = next(iter(dataloader))  # grab one batch
    print(f'Batch image dims {images.shape}, label dims {labels.shape}')        # [B, C, H, W], [B]
    print(f'Batch labels: {labels}')

def show_data_samples(dataset: torch.utils.data.Dataset):    
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i, ax in enumerate(axes):
        img, label = dataset[i]
        ax.imshow(img.squeeze(), cmap='gray')  # squeeze drops the [1,H,W] channel dim; cmap='gray' since single-channel
        ax.set_title(label)
        ax.axis('off')
    plt.show(block=False)
    plt.pause(2)

def cals_pred_accuracy(preds: torch.tensor, labels: torch.tensor)->float: 
    i_preds = preds.argmax(axis=1) # max logits indicates the preditced class, compare to label
    acc = (i_preds == labels).float().mean()
    return acc

data_path = 'D:\\GIT\\data'
train_dataset  = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())

is_ovrefit = False

if is_ovrefit:
    batch_size = 8

    n_batches = 2
    n_sub_samples = n_batches*batch_size
    sub_set = torch.utils.data.Subset(dataset=train_dataset, indices=list(range(n_sub_samples)))
    train_data_loader = torch.utils.data.DataLoader(dataset=sub_set, batch_size=batch_size, shuffle=False)
    test_data_loader = None
    d = 512
else:
    # full sized trining
    batch_size = 128
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset  = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    d = 64

# # inspect the data
# inspect_dataset(dataset=dataset)    # dataset
# inspect_dataloader(data_loader)
# show_data_samples(dataset=sub_set)

# # check labels disribution
# print(dataset.targets[:n_sub_samples])

img, label = train_dataset[0]
n_classes= len(train_dataset.classes)
n_enc_layers = 3

c, h, w =  img.shape
img_min_dim = min(h, w)
n_min_pathes = 36
n_path_dim = int(math.sqrt(n_min_pathes))
patch_size = max(img_min_dim // n_path_dim, 4) 
att_encoder = Encoder(n=n_enc_layers,d=d)
mlp_head = nn.Linear(in_features=d,out_features=n_classes)
vision_emb = VisionEmbeddings(d=d, img_size=img_min_dim, c=c, patch_size=patch_size)
ve = VisionEncoder(vision_emb=vision_emb, att_encoder=att_encoder, mlp_head=mlp_head)

n_epoch = 50
lr = 1e-4 # 1e-3 stuck att acc 0.25
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(lr=lr, params=ve.parameters())
# Train loop
n_train_batches = len(train_data_loader)
n_test_batches = len(test_data_loader)
targ_acc, min_loss = 0.95, 1e-3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device {device}')
ve.to(device)
for i_epoch in range(n_epoch):
    # Go thorugh epochs
    ve.train()
    loss_epoch = 0
    acc_batch = []
    for imgs, labels in  train_data_loader:
        # go torhoug all batches
        imgs, labels = imgs.to(device), labels.to(device)
        preds = ve(imgs)
        loss = loss_fn(preds, labels)
        ve.zero_grad()
        loss.backward()
        opt.step()

        loss_epoch += loss.item()
        acc_batch.append(cals_pred_accuracy(preds=preds, labels=labels))
    # ve.eval()
    # run evalution here if exist
    loss_epoch /= n_train_batches
    acc_epoch = torch.tensor(acc_batch).mean().item()
    status_s = f'Epoch {i_epoch+1}/{n_epoch}, Train Loss {loss_epoch:.2f}, Acc {acc_epoch:.2f}'
    
    if test_data_loader is not None:
        # run eval on test dataset
        loss_epoch = 0
        acc_batch = []
        with torch.no_grad():
            ve.eval()
            for imgs, labels in test_data_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                # go torhoug all batches
                preds = ve(imgs)
                loss = loss_fn(preds, labels)

                loss_epoch += loss.item()
                acc_batch.append(cals_pred_accuracy(preds=preds, labels=labels))
            # ve.eval()
            # run evalution here if exist
            loss_epoch /= n_test_batches
            acc_epoch = torch.tensor(acc_batch).mean().item()
            status_s += (f' Eval Loss {loss_epoch:.2f}, Acc {acc_epoch:.2f}')
    # Early stopping
    if acc_epoch >= targ_acc:
        print(f'Target accracy reached {acc_epoch :.2f}, stopping')
        break
    if loss_epoch <= min_loss:
        print(f'Target loss reached {loss_epoch :.2f}, stopping')
        break
    print(status_s)
