
import torch

def calculate_zoltar(Ef, Ep, Nf, Np):
    return torch.div(Ef, Ef + Nf + Ep + torch.div(10000 * Nf * Ep, Ef))