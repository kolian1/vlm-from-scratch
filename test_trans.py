from typing import List
import torch
from torch import optim
from torch import nn

from transformer import Transformer

class Tokeniser():
    '''
    Endcode chacter to their ASCII value, and decode back
    Add special vaues for BOS, EOS, PAD
    '''
    max_char_id = 127 # maximal value for encoide chaarcter

    def __init__(self, max_len:int=0):
        self.max_len = max_len-2 # the 2 is for BOS, EOS
        
        self.max_enc_id = self.max_char_id+ 1 # max value used by encoder
        self.PAD = self.max_enc_id
        self.max_enc_id += 1
        self.BOS = self.max_enc_id
        self.max_enc_id += 1
        self.EOS = self.max_enc_id

    def encode(self, line: str)->List[int]:
        enc_line = [ord(c) for c in line]
        n_line = len(line)
        n_pad = 0
        if self.max_len > 0:
            assert n_line <= self.max_len, f'line length {n_line} exceeds limit of {self.max_len}'
            # padding is needed
            n_pad = self.max_len - n_line
            
        pad = [self.PAD]*n_pad
        return [self.BOS] + enc_line + [self.EOS]+ pad
    
    def decode(self, tockens: List[int])->str:
        chars = [chr(toc) for toc in tockens if toc <= self.max_char_id]
        line = ''.join(chars)
        return line

def eval_model(model, toc, x, y):
    model.eval()
    with torch.no_grad():
        dec_input = y[:, :-1]
        dec_target = y[:, 1:]
        pred = model(x, dec_input)
        pred_ids = pred.argmax(dim=-1)          # [N, max_len-1], aligned with dec_target

        n_correct, n_total = 0, 0
        pred_lines, target_lines = [], []
        for i in range(pred_ids.shape[0]):
            p_line = toc.decode(pred_ids[i].tolist())
            t_line = toc.decode(dec_target[i].tolist())
            pred_lines.append(p_line)
            target_lines.append(t_line)
            for a, b in zip(pred_ids[i].tolist(), dec_target[i].tolist()):
                if b == toc.PAD:
                    continue          # don't count padding toward accuracy
                n_total += 1
                n_correct += int(a == b)

    acc = n_correct / (n_total + 1e-9)
    return acc, '\n'.join(target_lines), '\n'.join(pred_lines)


def toy_train():
    '''
    **Train a tiny demo:** 
    wire the training loop (CrossEntropyLoss, teacher-forced shifted-right target, causal mask) → 
    overfit a toy seq task (e.g. copy/reverse) → **loss→0**. 
    Proves the from-scratch build works end-to-end + closes the training-loop hands-on gap
    '''

    # As we plan to overfit I assume we do not need a long input

    x_text = '''
The Little Engine That Could is an American folktale and story to teach children the value of optimism and hard work. 
It is best known for its signature motif: "I think I can!"
The story originated in the early 20th century being retold by various authors, including Mary C. Jacobs. 
It was first referred to as its well known title in a 1920 edition published within the My Book House series. 
The most widely known version by Arnold "Watty Piper" Munk was published in 1930 by Platt & Munk. 
The 1930 version entered the American public domain on January 1, 2026.
Plot
In the tale, a long train must be pulled over a high mountain after its locomotive breaks down. 
Larger locomotives, treated anthropomorphically, are asked to pull the train; 
for various reasons, they refuse because they think they are too important (or in the old engines case, because he cant due to his age). 
The request is sent to a small engine, who agrees to try. 
Despite the steep climb and heavy load, the engine slowly succeeds in pulling the train over the mountain while repeating the motto: "I think I can".
    '''
    # lets revert ech line, keepin order the same, and see if this can be overfitted
    x_lines =x_text.split('\n')
    y_lines = [line[::-1] for line in x_lines] # is there a better way to revert strings?
    y_text = '\n'.join(y_lines)
    n_x_chars = len(x_text)

    max_line_length = max([len(line) for line in x_lines])
    n_lines = len(x_lines)
    max_len = 2+max_line_length
    n = 4
    d = int(2**8)


    # tockenise x and y- wonder how will tokenisation deal with non words in y
    toc = Tokeniser(max_len=max_len)
    x_tokens = [toc.encode(line) for line in x_lines]
    x_tokens = torch.tensor(x_tokens)
    y_tokens = [toc.encode(line) for line in y_lines]
    y_tokens = torch.tensor(y_tokens)

    pad_line = torch.tensor([toc.encode('')])
    
    vocab_size = int(toc.max_enc_id+1)
    n_batch_lines = 4 # no of lines in batch
    
    n_last_batch_lines = n_lines % n_batch_lines
    n_pad_lines = n_batch_lines - n_last_batch_lines if n_last_batch_lines else 0
    pad_lines= pad_line.repeat(n_pad_lines, 1)
    # break x and y to batches
    # how to make it of shape [n_batch_lines, max_line_length]
    x_flat = torch.cat([x_tokens, pad_lines], dim=0)
    y_flat = torch.cat([y_tokens, pad_lines], dim=0)

    n_batches = x_flat.shape[0]//n_batch_lines

    n_epoch = 500
    n_print = min(max(1, int(n_epoch/10)), 10)
    lr = 1e-3
    # train tranformer
    model = Transformer(vocab_size=vocab_size, max_len=max_len, d=d, n=n)
    loss_fn = nn.CrossEntropyLoss(ignore_index=toc.PAD)
    opt = optim.Adam(model.parameters(), lr=lr)
    stop_acc = 0.975

    for i_epoch in range(n_epoch):
        # how to shuffle x, y to keep them synced
        idx = torch.randperm(x_flat.shape[-2])  # shape[-2] = number of rows
        x_flat = x_flat[idx]
        y_flat = y_flat[idx]

        x = x_flat.reshape(n_batches, n_batch_lines, max_len)
        y = y_flat.reshape(n_batches, n_batch_lines, max_len)

        dec_input =  y[:, :,:-1] # drop EOS/PAD
        dec_target = y[:, :,1:] # drop BOS

        loss_val = 0
        model.train()
        for i_batch in range(n_batches):
            pred = model(x[i_batch], dec_input[i_batch]) # [b, line, vocab] # [i_batch] or [i_batch, :, :]
            loss = loss_fn(pred.permute(0,2,1), dec_target[i_batch])

            model.zero_grad()
            loss.backward()
            opt.step()
            # model. backpropogate or opt?
            
            loss_val += loss.item()
            # acc decoded pred vs decoded y

        # report loss
        # norm loss by no of samples to haev same values for different sizes
        if i_epoch%n_print == 0:
            loss_val /= n_x_chars
            acc, y_dec_text, pred_dec_test = eval_model(model=model, toc=toc, x=x_flat, y=y_flat)
            print(f'Epoch {i_epoch+1}/{n_epoch}, Train loss:{loss_val:.5f}, Train Acc {acc:.4f}')

            if acc > stop_acc:
                print(f'Target Accuracy {stop_acc} exceeded, stopping early')
                break
    
    # test tranfomrmer on x and y
    # lates model evaluation
    acc, y_dec_text, pred_dec_test = eval_model(model=model, toc=toc, x=x_flat, y=y_flat)
    print(f'GT text\n{y_dec_text}')
    print(f'Pred text\n{pred_dec_test}')

toy_train()