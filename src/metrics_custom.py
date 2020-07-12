import torch

from src.config import LABELS, RED, BLACK

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

name2id = {v:k for k,v in enumerate(LABELS+[RED, BLACK])}  # {WHITE:0, BLUE:1}
void_codes_red = name2id[RED]
void_codes_black = name2id[BLACK]


def acc_satellite(input, target):
    target = target.squeeze(1)
    mask = target != void_codes_red
    mask = target != void_codes_black
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


def acc_weakly(input, target):
    bs, ncolors, width, height = input.shape
    input = input.reshape(bs, ncolors, -1)
    input = input.argmax(dim=1)  # shape: (bs, pixel)

    mat = torch.zeros(input.shape).to(device)
    for batch_idx in range(bs):
        for i in range(ncolors):
            if target[batch_idx][i] == 0:
                continue
            mat[batch_idx][input[batch_idx]==i] = 1
    return mat.float().mean()
