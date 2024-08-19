
import torch

def calculate_zoltar(Ef, Ep, Nf, Np):
    epsilon = 1e-10  # Small value to prevent division by zero
    return Ef / (Ef + Nf + Ep + (10000 * Nf * Ep / (Ef + epsilon)))