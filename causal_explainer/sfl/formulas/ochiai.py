import torch

def calculate_ochiai(Ef, Ep, Nf, Np):
    numerator = Ef
    denominator = torch.sqrt((Ef + Nf) * (Ef + Ep))
    return torch.div(numerator, denominator)