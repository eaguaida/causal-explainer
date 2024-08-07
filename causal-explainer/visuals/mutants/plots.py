import torch
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

class SFLVisualizer:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.tensor_imshow = utils.tensor_imshow
        
    def denormalize(self, tensor):
        mean = self.mean.view(1, 3, 1, 1).to(tensor.device)
        std = self.std.view(1, 3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def visualize_single_mutant(self, mutants, pass_fail_list, index):
        mutant_image = mutants[index].detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(5, 5))
        plt.imshow(mutant_image)
        plt.title(f"Mutant {index+1}\n{pass_fail_list[index]}")
        plt.axis('off')
        plt.show()

    def interactive_mutant_visualization(self, sampled_tensor, pass_fail_list):
        mutants_denorm = self.denormalize(sampled_tensor)
        num_mutants = sampled_tensor.shape[0]
        
        interact(self.visualize_single_mutant,
                 mutants=fixed(mutants_denorm),
                 pass_fail_list=fixed(pass_fail_list),
                 index=IntSlider(min=0, max=num_mutants-1, step=1, description='Mutant Index:'))

    def plot_saliency_map(self, img, saliency_map, target_class):
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title("Original Image")
        self.tensor_imshow(img[0])
        
        # Saliency map
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title(f"Saliency Map for {self.get_class_name(target_class)}")
        self.tensor_imshow(img[0])
        plt.imshow(saliency_map.cpu().numpy(), cmap='jet', alpha=0.5)
        
        plt.tight_layout()
        plt.show()