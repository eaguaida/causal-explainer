import sys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np
import argparse

# Ensure RISE-SFL is in the Python path
rise_sfl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'causal-explainer'))
sys.path.append(rise_sfl_path)

from utils.utils import *
from masker.generation import SFL
from masker.batch import SFL_batch
from sfl_techniques.relevance_score import RelevanceScore
from visuals.saliency_maps.saliency import SaliencyMapVisualizer

cudnn.benchmark = True

def batch(model, image_folder, N, s=8, p1=0.2, save_maps=False):
    input_size = (224, 224)
    sfl_batch = SFL_batch(model, input_size)
    
    # Generate batch images
    masks, sampled_tensors, target_list = sfl_batch.generate_batch_images(image_folder, N, s, p1)
    
    # Process images and calculate confidence scores
    confidence_scores = np.zeros((len(sampled_tensors), N))
    with torch.no_grad():
        for i in range(len(sampled_tensors)):
            for j in range(N):
                single_image = sampled_tensors[i, j].unsqueeze(0)
                output = model(single_image)
                confidence_scores[i, j] = torch.max(output, dim=1).values.item() * 100
    print('Calculating relevance scores, please wait... This may take a while.')
    # Calculate relevance scores
    relevance_score_calculator = RelevanceScore(device='cuda')
    pixel_datasets = []
    ochiai_array, zoltar_array, tarantula_array, wong1_array = [], [], [], []
    
    for i in range(len(sampled_tensors)):
        pixel_dataset, ochiai, tarantula, zoltar, wong1 = relevance_score_calculator.run(
            confidence_scores[i], sampled_tensors[i], masks[i], N)
        pixel_datasets.append(pixel_dataset)
        ochiai_array.append(ochiai)
        zoltar_array.append(zoltar)
        tarantula_array.append(tarantula)
        wong1_array.append(wong1)
    
    # Visualize or save saliency maps
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', 'jpeg'))]
    if save_maps:
        saliency_folder = "saliency_maps"
        os.makedirs(saliency_folder, exist_ok=True)
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        saliency_map = SaliencyMapVisualizer(image_path)
        if save_maps:
            output_path = os.path.join(saliency_folder, f"saliency_map_{i}.png")
            saliency_map.save_pixel_scores(pixel_datasets[i], output_path, ins='')
        else:
            saliency_map.visualize_pixel_scores(pixel_datasets[i], ins='')
    
    return {
        'masks': masks,
        'sampled_tensors': sampled_tensors,
        'target_list': target_list,
        'confidence_scores': confidence_scores,
        'pixel_datasets': pixel_datasets,
        'ochiai_array': ochiai_array,
        'zoltar_array': zoltar_array,
        'tarantula_array': tarantula_array,
        'wong1_array': wong1_array
    }

def main():
    parser = argparse.ArgumentParser(description='Run batch processing')
    parser.add_argument('image_folder', type=str, help='Path to the image folder')
    parser.add_argument('N', type=int, help='Number of mutations')
    parser.add_argument('--s', type=int, default=8, help='Mask size (default: 8)')
    parser.add_argument('--p1', type=float, default=0.2, help='Probability (default: 0.2)')
    args = parser.parse_args()

    # Load the default model (ResNet50)
    model = models.resnet50(True)
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval().cuda()
    for p in model.parameters():
        p.requires_grad = False

    results = batch(model, args.image_folder, args.N, args.s, args.p1, save_maps=True)
    if results:
        print("Processing complete. Saliency maps have been saved in the 'saliency_maps' folder.")
    else:
        print("Processing failed. Please check the input parameters and try again.")

if __name__ == "__main__":
    main()