import torch

def calculate_tarantula(Ef, Ep, Nf, Np):
    numerator = torch.div(Ef, Ef + Nf)
    denominator = numerator + torch.div(Ep, Ep + Np)
    return torch.div(numerator, denominator)