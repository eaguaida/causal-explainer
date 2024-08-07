import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode
from tqdm import tqdm
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils

class SFL:
    def __init__(self, model, input_size):
        self.model = model
        self.input_size = torch.tensor(input_size, device='cuda')
        self.device = torch.device('cuda')

    def generate_support_masks(self, N, s, p1, input_size=(224, 224)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Calculate cell size and up size
        cell_size = torch.ceil(torch.tensor(self.input_size).float() / s)
        up_size = ((s + 1) * cell_size).int()

        # Create random grid
        grid = (torch.rand(N, s, s) < p1).float().to(self.device)

        # Initialize support masks
        support_masks = torch.empty(N, 1, *input_size, device=self.device)

        for i in range(N):
            # Random shifts
            x = torch.randint(0, int(cell_size[0]), (1,))
            y = torch.randint(0, int(cell_size[1]), (1,))

            # Resize grid
            resized_grid = resize(grid[i].unsqueeze(0), tuple(up_size.int().tolist()), 
                                  interpolation=InterpolationMode.BILINEAR)

            # Crop and assign to support masks
            support_masks[i, 0] = resized_grid[0, 
                                               x:x + input_size[0], 
                                               y:y + input_size[1]]
        return support_masks

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
    
    @torch.no_grad()
    def generate_mutants_batch(self, img_path, N, s, p1, target_class, batch_size=50):   
        self.path = img_path
        img_tensor = utils.read_tensor(img_path).to(self.device).requires_grad_(False)
        _, channels, height, width = img_tensor.shape
        self.masks = torch.empty((N, 1, height, width), device=self.device)
        sampled_tensor = torch.empty((N, channels, height, width), device=self.device)
        
        for start_idx in tqdm(range(0, N, batch_size), total=(N + batch_size - 1) // batch_size, ascii="░▒█"):
            end_idx = min(start_idx + batch_size, N)
            batch_size_current = end_idx - start_idx
            
            batch_masks = torch.empty((batch_size_current, 1, height, width), device=self.device)
            found_flags = torch.zeros(batch_size_current, dtype=torch.bool, device=self.device)
            
            while not found_flags.all():
                # Generate masks for all unfound samples in the batch
                unfound_indices = torch.where(~found_flags)[0]
                pass_indices = unfound_indices[unfound_indices % 2 == 0]
                fail_indices = unfound_indices[unfound_indices % 2 == 1]
                
                if len(pass_indices) > 0:
                    batch_masks[pass_indices] = self.generate_support_masks(len(pass_indices), s, 0.2, input_size=(height, width))
                if len(fail_indices) > 0:
                    batch_masks[fail_indices] = self.generate_support_masks(len(fail_indices), s, 0.8, input_size=(height, width))
                
                # Apply masks to all images in the batch simultaneously
                masked_images = torch.mul(batch_masks, img_tensor.expand(batch_size_current, -1, -1, -1))
                
                # Perform inference on the entire batch at once
                outputs = self.model(masked_images)
                top_classes = outputs.argmax(dim=1)
                
                # Check conditions for all samples in the batch
                pass_condition = (top_classes == target_class) & (~found_flags) & (torch.arange(batch_size_current, device=self.device) % 2 == 0)
                fail_condition = (top_classes != target_class) & (~found_flags) & (torch.arange(batch_size_current, device=self.device) % 2 == 1)
                
                new_found = pass_condition | fail_condition
                found_flags |= new_found
                
                # Update masks and sampled tensor for newly found samples
                found_indices = start_idx + torch.where(new_found)[0]
                self.masks[found_indices] = batch_masks[new_found].clone().detach().requires_grad_(False)
                sampled_tensor[found_indices] = masked_images[new_found].clone().detach().requires_grad_(False)
        
        return self.masks, sampled_tensor