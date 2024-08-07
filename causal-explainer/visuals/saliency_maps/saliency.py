import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import os
import torch
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfl_techniques import sfl

class SaliencyMapVisualizer:
    def __init__(self, img_path, device='cuda'):
        # Initialize and compute the relevance scores
        self.original_image = Image.open(img_path).resize((224, 224)).convert('L')
        self.original_image_array = np.array(self.original_image)

    def visualize_pixel_scores(self, dataset, ins=''):
        H, W = (224,224)
        scores = {
            'Ep': np.zeros((H, W)),
            'Ef': np.zeros((H, W)),
            'Np': np.zeros((H, W)),
            'Nf': np.zeros((H, W)),
            'ochiai': np.zeros((H, W)),
            'tarantula': np.zeros((H, W)),
            'zoltar': np.zeros((H, W)),
            'wong1': np.zeros((H, W))
        }
        
        for pixel in dataset:
            i, j = pixel['position']
            for score_type in scores.keys():
                scores[score_type][i, j] = pixel[score_type]
        
        # Invert values for ochiai and tarantula
        scores['ochiai'] = 1 - scores['ochiai']
        scores['tarantula'] = 1 - scores['tarantula']
        scores['zoltar']  = 1 - scores['zoltar']
        scores['wong1']  = 1 - scores['wong1']
        
        if ins.lower() == 'all':
            plot_scores = list(scores.keys())
        elif ins.lower() in scores:
            plot_scores = [ins.lower()]
        else:  # Default case, including when ins is empty or 'None'
            plot_scores = ['tarantula', 'ochiai', 'zoltar', 'wong1']
        # Set up the plot
        num_plots = len(plot_scores)
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(5, 4))  # 50% smaller than the original (10, 8)
        else:
            rows = (num_plots + 3) // 4  # Round up to the nearest multiple of 4
            cols = min(4, num_plots)
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
            if num_plots > 1:
                axes = axes.flatten()

        fig.suptitle("Statistical Fault Localisation - Saliency Maps", fontsize=16)
        cmap = plt.get_cmap('jet')

        # Plot each selected score type
        for idx, score_type in enumerate(plot_scores):
            if num_plots == 1:
                current_ax = ax
            else:
                current_ax = axes[idx]
            
            # Create the heatmap
            current_ax.imshow(self.original_image_array, cmap='gray', alpha=1)
            im = current_ax.imshow(scores[score_type], cmap=cmap, alpha=0.5)
            current_ax.set_title(score_type.capitalize())
            current_ax.axis('off')
            
            # Add colorbar
            divider = make_axes_locatable(current_ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        # Remove any unused subplots
        if num_plots > 1:
            for idx in range(num_plots, len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
        