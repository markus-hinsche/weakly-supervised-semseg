import torch

THRESH_LOWER_CLIP_PROBS = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeakCrossEntropy:
    """Cross Entropy for semantically segmented images based on weak labels.

    Weak labels are vector of color flags.
    Each flags represents whether a certain color exists in the image or not.
    """

    def __init__(self, axis=1):
        self.axis = axis
        assert axis == 1
        self.one_tensor = torch.Tensor([1.0]).to(DEVICE)

    def __call__(self, pred, target):
        """
        pred: predictions of shape (bs,ncolors,width,height)

        target: of shape (bs, ncolors)
                Example tensor([[1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 0.]])
        """
        bs, ncolors, width, height = pred.shape
        assert target.shape[0] == bs

        ncolors_in_target = target.shape[1]
        if ncolors_in_target != ncolors:
            target_to_concat = torch.tensor(bs, ncolors - ncolors_in_target)
            target = torch.cat((target, target_to_concat), dim=1)

        pred_ = pred.softmax(dim=1)

        # Flatten the input
        pred_ = pred_.reshape(bs, ncolors, -1)  # shape (bs, ncolors, width*height)

        if torch.isnan(pred_).any():
            raise Exception("input is nan: " + str(pred_))

        target_mask = (
            target.repeat(width * height, 1, 1).transpose(0, 1).transpose(1, 2)
        )  # shape (bs, ncolors, w*h)

        sums_prob_y_1 = (pred_ * target_mask).sum(axis=1)  # shape (bs, width*height)

        # Clip low probabilities
        sums_prob_y_1[sums_prob_y_1 < THRESH_LOWER_CLIP_PROBS] = THRESH_LOWER_CLIP_PROBS

        item_losses = sums_prob_y_1.log() * -1.0  # shape (bs, width*height)

        assert item_losses.shape == (bs, width * height)
        assert not torch.isinf(item_losses).any()

        loss = item_losses.mean()

        if torch.isnan(input).any():
            raise Exception("loss is nan: " + str(sums_prob_y_1))

        return loss
