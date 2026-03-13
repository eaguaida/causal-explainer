import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from causal_explainer.utils import get_class_name, get_device

HW = 224 * 224
n_classes = 1000
KLEN = 11
NSIG = 5


def gkern(klen, nsig):
    """Returns a Gaussian kernel array."""
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, nsig)
    return k.astype('float32')


KERN = gkern(KLEN, NSIG)


def blur_image(x):
    """Blur image using Gaussian kernel (PyTorch only)."""
    kern_torch = torch.from_numpy(KERN).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    return torch.nn.functional.conv2d(x, kern_torch, padding=KLEN // 2, groups=3)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric:
    def __init__(self, model, mode, step, substrate_fn, device=None):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            device: torch device (auto-detected if None).
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.device = device or get_device()

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        img_tensor = img_tensor.to(self.device)
        pred = self.model(img_tensor)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]

        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)

        for i in range(n_steps + 1):
            pred = self.model(start)
            pr, cl = torch.topk(pred, 2)
            scores[i] = pred[0, c].cpu().numpy()

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start_np = start.cpu().numpy().reshape(1, 3, HW)
                finish_np = finish.cpu().numpy().reshape(1, 3, HW)
                start_np[0, :, coords] = finish_np[0, :, coords]
                start = torch.from_numpy(start_np.reshape(1, 3, 224, 224)).to(self.device)

            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
        return scores
