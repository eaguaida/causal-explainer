import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


class SaliencyMapVisualizer:
    def __init__(self, img_path):
        self.original_image = Image.open(img_path).convert('L')

    def _infer_shape(self, dataset):
        if not dataset:
            raise ValueError("Dataset must not be empty.")

        max_row = max(pixel['position'][0] for pixel in dataset)
        max_col = max(pixel['position'][1] for pixel in dataset)
        return max_row + 1, max_col + 1

    def _build_score_grids(self, dataset):
        H, W = self._infer_shape(dataset)
        scores = {
            'Ep': np.zeros((H, W)),
            'Ef': np.zeros((H, W)),
            'Np': np.zeros((H, W)),
            'Nf': np.zeros((H, W)),
            'ochiai': np.zeros((H, W)),
            'tarantula': np.zeros((H, W)),
            'zoltar': np.zeros((H, W)),
            'wong1': np.zeros((H, W)),
        }

        for pixel in dataset:
            i, j = pixel['position']
            for score_type in scores:
                scores[score_type][i, j] = pixel[score_type]

        scores['ochiai'] = 1 - scores['ochiai']
        scores['tarantula'] = 1 - scores['tarantula']
        scores['zoltar'] = 1 - scores['zoltar']
        scores['wong1'] = 1 - scores['wong1']

        return scores

    def _select_scores(self, scores, ins):
        key = ins.lower()
        if key == 'all':
            return list(scores.keys())
        if key in scores:
            return [key]
        return ['tarantula', 'ochiai', 'zoltar', 'wong1']

    def _plot_scores(self, scores, plot_scores):
        height, width = next(iter(scores.values())).shape
        original_image_array = np.array(self.original_image.resize((width, height)))
        num_plots = len(plot_scores)
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            axes_list = [ax]
        else:
            rows = (num_plots + 3) // 4
            cols = min(4, num_plots)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes_list = axes.flatten() if num_plots > 1 else [axes]

        fig.suptitle("Statistical Fault Localisation - Saliency Maps", fontsize=16)
        cmap = plt.get_cmap('jet')

        for idx, score_type in enumerate(plot_scores):
            current_ax = axes_list[idx]
            current_ax.imshow(original_image_array, cmap='gray', alpha=1)
            im = current_ax.imshow(scores[score_type], cmap=cmap, alpha=0.5)
            current_ax.set_title(score_type.capitalize())
            current_ax.axis('off')

            divider = make_axes_locatable(current_ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        if num_plots > 1:
            for idx in range(num_plots, len(axes_list)):
                fig.delaxes(axes_list[idx])

        plt.tight_layout()
        return fig

    def visualize_pixel_scores(self, dataset, ins=''):
        scores = self._build_score_grids(dataset)
        plot_scores = self._select_scores(scores, ins)
        self._plot_scores(scores, plot_scores)
        plt.show()

    def save_pixel_scores(self, dataset, output_path, ins=''):
        scores = self._build_score_grids(dataset)
        plot_scores = self._select_scores(scores, ins)
        fig = self._plot_scores(scores, plot_scores)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
