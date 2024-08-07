import torch
import numpy as np
import math
class FaultLocalizationMetrics:
    @staticmethod
    def calculate_ochiai(Ef, Ep, Nf, Np):
        numerator = Ef
        denominator = torch.sqrt((Ef + Nf) * (Ef + Ep))
        return torch.div(numerator, denominator)

    @staticmethod
    def calculate_tarantula(Ef, Ep, Nf, Np):
        numerator = torch.div(Ef, Ef + Nf)
        denominator = numerator + torch.div(Ep, Ep + Np)
        return torch.div(numerator, denominator)

    @staticmethod
    def calculate_zoltar(Ef, Ep, Nf, Np):
        return torch.div(Ef, Ef + Nf + Ep + torch.div(10000 * Nf * Ep, Ef))

    @staticmethod
    def calculate_wong1(Ef, Ep, Nf, Np):
        return Ef - Ep


class RelevanceScore:
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()

    def reset(self):
        self.Ep = None
        self.Ef = None
        self.Np = None
        self.Nf = None
        self.scores_dict = {}
        self.dataset = []
        self.ochiai_array = None
        self.tarantula_array = None
        self.zoltar_array = None
        self.wong1_array = None

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
        badscalar = good_scores.view(m, 1, 1, 1)

        e_pass_tensors = torch.mul(executed_tensors[pass_indices], goodscalar)
        e_fail_tensors = torch.mul(executed_tensors[fail_indices], badscalar)
        n_pass_tensors = torch.mul(not_executed_tensors[pass_indices], goodscalar)
        n_fail_tensors = torch.mul(not_executed_tensors[fail_indices], badscalar)

        self.Ep = e_pass_tensors.sum(dim=0)
        self.Ef = e_fail_tensors.sum(dim=0)
        self.Np = n_pass_tensors.sum(dim=0)
        self.Nf = n_fail_tensors.sum(dim=0)

    def calculate_all_scores(self,confidence_scores, img, masks, N):
        self.calculate_relevance_scores(confidence_scores,img, masks, N)
        ochiai_scores = FaultLocalizationMetrics.calculate_ochiai(self.Ef, self.Ep, self.Nf, self.Np)
        tarantula_scores = FaultLocalizationMetrics.calculate_tarantula(self.Ef, self.Ep, self.Nf, self.Np)
        zoltar_scores = FaultLocalizationMetrics.calculate_zoltar(self.Ef, self.Ep, self.Nf, self.Np)
        wong1_scores = FaultLocalizationMetrics.calculate_wong1(self.Ef, self.Ep, self.Nf, self.Np)

        self.ochiai_array = ochiai_scores.detach().cpu().numpy().max() - ochiai_scores.detach().cpu().numpy()
        self.tarantula_array = tarantula_scores.detach().cpu().numpy().max() - tarantula_scores.detach().cpu().numpy()
        self.zoltar_array = zoltar_scores.detach().cpu().numpy().max() -zoltar_scores.detach().cpu().numpy()
        self.wong1_array = wong1_scores.detach().cpu().numpy().max() - wong1_scores.detach().cpu().numpy()

        self.scores_dict = {
            'Ep': self.Ep,
            'Ef': self.Ef,
            'Np': self.Np,
            'Nf': self.Nf,
            'ochiai': ochiai_scores,
            'tarantula': tarantula_scores,
            'zoltar': zoltar_scores,
            'wong1': wong1_scores
        }

        return self.scores_dict
    
    def create_pixel_dataset(self, img_shape):
        H, W = img_shape[-2:]
        for i in range(H):
            for j in range(W):
                pixel_data = {
                    'position': (i, j),
                    'Ep': self.scores_dict['Ep'][0, i, j].item(),
                    'Ef': self.scores_dict['Ef'][0, i, j].item(),
                    'Np': self.scores_dict['Np'][0, i, j].item(),
                    'Nf': self.scores_dict['Nf'][0, i, j].item(),
                    'ochiai': self.scores_dict['ochiai'][0, i, j].item(),
                    'tarantula': self.scores_dict['tarantula'][0, i, j].item(),
                    'zoltar': self.scores_dict['zoltar'][0, i, j].item(),
                    'wong1': self.scores_dict['wong1'][0, i, j].item()
                }
                self.dataset.append(pixel_data)
        return self.dataset

    def run(self, confidence_scores, img, masks, N):
        self.calculate_all_scores(confidence_scores,img, masks, N)
        dataset = self.create_pixel_dataset(img.shape)
        result = dataset.copy()  # Create a copy of the dataset
        ochiai_array = self.ochiai_array.copy()
        tarantula_array = self.tarantula_array.copy()
        zoltar_array = self.zoltar_array.copy()
        wong1_array = self.wong1_array.copy()
        self.reset()
        return result, ochiai_array, tarantula_array, zoltar_array, wong1_array

