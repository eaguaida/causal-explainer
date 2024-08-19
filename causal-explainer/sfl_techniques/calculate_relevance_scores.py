import torch
import math

def calculate_relevance_scores(self, confidence_scores, sampled_tensor, mask, N):
    device = self.device
    sampled_tensor = sampled_tensor.to(device)
    _, _, H, W = sampled_tensor.shape
    
    all_indices = torch.arange(N, device=device)
    pass_indices = all_indices[all_indices % 2 != (N % 2)]
    fail_indices = all_indices[all_indices % 2 == (N % 2)]
    
    confidence_scores = torch.tensor(confidence_scores, dtype=torch.float32, device=device)
    good_scores, fail_scores = confidence_scores[::2], confidence_scores[1::2]
    
    m = math.ceil(N/2)
    goodscalar, badscalar = good_scores.view(m, 1, 1, 1), fail_scores.view(m, 1, 1, 1)
    
    executed_tensors = mask
    not_executed_tensors = 1 - mask
    
    e_pass_tensors = executed_tensors[pass_indices] * goodscalar
    e_fail_tensors = executed_tensors[fail_indices] * badscalar
    n_pass_tensors = not_executed_tensors[pass_indices] * goodscalar
    n_fail_tensors = not_executed_tensors[fail_indices] * badscalar
    self.Ep = e_pass_tensors.sum(dim=0)
    self.Ef = e_fail_tensors.sum(dim=0)
    self.Np = n_pass_tensors.sum(dim=0)
    self.Nf = n_fail_tensors.sum(dim=0)

    