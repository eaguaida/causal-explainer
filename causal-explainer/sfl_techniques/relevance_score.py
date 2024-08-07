import torch
from .initialize import initialize
from .reset import reset
from .calculate_relevance_scores import calculate_relevance_scores
from .calculate_all_scores import calculate_all_scores
from .create_pixel_dataset import create_pixel_dataset
from .run import run



class RelevanceScore:
    def __init__(self, device='cuda'):
        self.device = device
        initialize(self)

    def reset(self):
        reset(self)

    def calculate_relevance_scores(self, confidence_scores, sampled_tensor, mask, N):
        return calculate_relevance_scores(self, confidence_scores, sampled_tensor, mask, N)

    def calculate_all_scores(self, confidence_scores, img, masks, N):
        return calculate_all_scores(self, confidence_scores, img, masks, N)

    def create_pixel_dataset(self, img_shape):
        return create_pixel_dataset(self, img_shape)

    def run(self, confidence_scores, img, masks, N):
        return run(self, confidence_scores, img, masks, N)