import torch
import math

def calculate_relevance_scores(self, confidence_scores, sampled_tensor, mask, N):
    sampled_tensor = sampled_tensor.to(self.device)

    _, C, H, W = sampled_tensor.shape
    all_indices = torch.arange(N, device=self.device)
    pass_indices = all_indices[all_indices % 2 == (0 if N % 2 == 0 else 1)]
    fail_indices = all_indices[all_indices % 2 != (0 if N % 2 == 0 else 1)]
    mask = mask.to(self.device)
    shape = (N, 1, H, W)
    tensor_ones = torch.ones(shape).to(self.device)

    executed_tensors = mask
    not_executed_tensors = tensor_ones - mask
    good_scores = confidence_scores[::2]  # Every 2nd element, starting from index 0
    fail_scores = confidence_scores[1::2]

    good_scores = torch.tensor(good_scores, dtype=torch.float32, device=self.device)
    fail_scores = torch.tensor(fail_scores, dtype=torch.float32, device=self.device)
    m = math.ceil(N/2)
    goodscalar = good_scores.view(m, 1, 1, 1)
    badscalar = fail_scores.view(m, 1, 1, 1)

    e_pass_tensors = torch.mul(executed_tensors[pass_indices], goodscalar)
    e_fail_tensors = torch.mul(executed_tensors[fail_indices], badscalar)
    n_pass_tensors = torch.mul(not_executed_tensors[pass_indices], goodscalar)
    n_fail_tensors = torch.mul(not_executed_tensors[fail_indices], badscalar)

    self.Ep = e_pass_tensors.sum(dim=0)
    self.Ef = e_fail_tensors.sum(dim=0)
    self.Np = n_pass_tensors.sum(dim=0)
    self.Nf = n_fail_tensors.sum(dim=0)