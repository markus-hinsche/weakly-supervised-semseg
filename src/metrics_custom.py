
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
    # print(mat)
    return mat.float().mean()
