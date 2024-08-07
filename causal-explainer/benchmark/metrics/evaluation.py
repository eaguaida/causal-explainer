from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf

from utils import *
HW = 224 * 224 # image area
n_classes = 1000
KLEN = 11
NSIG = 5 

def gkern(klen, nsig):
    """Returns a Gaussian kernel array."""
    inp = np.zeros((klen, klen))
    inp[klen//2, klen//2] = 1
    k = gaussian_filter(inp, nsig)
    return k.astype('float32')

# Pre-compute the kernel

KERN = gkern(KLEN, NSIG)

def blur_image(x, model_type='pytorch'):
    if model_type == 'pytorch':
        # PyTorch implementation
        kern_torch = torch.from_numpy(KERN).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        return torch.nn.functional.conv2d(x, kern_torch, padding=KLEN//2, groups=3)
    else:
        # TensorFlow implementation
        kern_tf = tf.convert_to_tensor(KERN)
        kern_tf = tf.expand_dims(tf.expand_dims(kern_tf, axis=-1), axis=-1)
        kern_tf = tf.repeat(kern_tf, 3, axis=2)  # Repeat for each channel
        
        # Ensure input has 3 channels
        if x.shape[-1] == 1:
            x = tf.repeat(x, 3, axis=-1)
        
        return tf.nn.depthwise_conv2d(x, kern_tf, strides=[1,1,1,1], padding='SAME')

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():
    def __init__(self, model, mode, step, substrate_fn, model_type='tensorflow'):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.model_type = model_type
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        if self.model_type == 'pytorch':
            img_tensor = img_tensor.cuda()
            pred = self.model(img_tensor)
            top, c = torch.max(pred, 1)
            c = c.cpu().numpy()[0]
        else:  # TensorFlow
            pred = self.model(img_tensor)
            top = tf.reduce_max(pred, axis=1)
            c = tf.argmax(pred, axis=1)[0].numpy()
            print(get_class_name(c))

        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone() if self.model_type == 'pytorch' else tf.identity(img_tensor)
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone() if self.model_type == 'pytorch' else tf.identity(img_tensor)

        scores = np.empty(n_steps + 1)
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        
        for i in range(n_steps + 1):
            if self.model_type == 'pytorch':
                pred = self.model(start)
                pr, cl = torch.topk(pred, 2)
            else:  # TensorFlow
                pred = self.model(start)
                pr, cl = tf.math.top_k(pred, k=2)

            scores[i] = pred[0, c].cpu().numpy() if self.model_type == 'pytorch' else pred[0, c].numpy()

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                if self.model_type == 'pytorch':
                    start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
                else:
                    start_np = start.numpy()
                    finish_np = finish.numpy()
                    start_np.reshape(1, 3, HW)[0, :, coords] = finish_np.reshape(1, 3, HW)[0, :, coords]
                    start = tf.convert_to_tensor(start_np)
                    
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
        return scores