import os
import torch
from tqdm import tqdm
from .generation import SFL
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
class SFL_batch(SFL):
    def __init__(self, model, input_size):
        super().__init__(model, input_size)

    def generate_batch_images(self, image_folder, N, s, p1, batch_size=50):
        # Get all image files from the folder
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_masks = []
        all_sampled_tensors = []
        target_list = []
        for img_file in tqdm(image_files, desc="Processing images"):
            # Load and preprocess the image using read_tensor
            img_path = os.path.join(image_folder, img_file)
            img_tensor = utils.read_tensor(img_path).to(self.device)
            with torch.no_grad():
                output = self.model(img_tensor)
                top_probabilities, top_classes = torch.topk(output, k=1, dim=1)
                top_pred_class = top_classes[0][0].item()

            target_class = top_pred_class
            target_list.append(target_class)
            # Process this image
            _, channels, height, width = img_tensor.shape
            masks = torch.empty((1, N, 1, height, width), device=self.device)
            sampled_tensor = torch.empty((1, N, channels, height, width), device=self.device)
            
            for start_idx in range(0, N, batch_size):
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
                        batch_masks[pass_indices] = self.generate_support_masks(len(pass_indices), s, p1, input_size=(height, width))
                    if len(fail_indices) > 0:
                        batch_masks[fail_indices] = self.generate_support_masks(len(fail_indices), s, (1-p1), input_size=(height, width))
                    
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
                    found_indices = torch.where(new_found)[0]
                    masks[0, start_idx + found_indices] = batch_masks[new_found].clone().detach().requires_grad_(False)
                    sampled_tensor[0, start_idx + found_indices] = masked_images[new_found].clone().detach().requires_grad_(False)
            
            all_masks.append(masks)
            all_sampled_tensors.append(sampled_tensor)

        # Stack results from all images
        combined_masks = torch.cat(all_masks, dim=0)
        combined_sampled_tensors = torch.cat(all_sampled_tensors, dim=0)
        return combined_masks, combined_sampled_tensors, target_list 