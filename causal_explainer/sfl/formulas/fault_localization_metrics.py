# fault_localization_metrics.py
from .ochiai import calculate_ochiai
from .tarantula import calculate_tarantula
from .zoltar import calculate_zoltar
from .wong1 import calculate_wong1

class FaultLocalizationMetrics:
    @staticmethod
    def calculate_ochiai(Ef, Ep, Nf, Np):
        return calculate_ochiai(Ef, Ep, Nf, Np)

    @staticmethod
    def calculate_tarantula(Ef, Ep, Nf, Np):
        return calculate_tarantula(Ef, Ep, Nf, Np)

    @staticmethod
    def calculate_zoltar(Ef, Ep, Nf, Np):
        return calculate_zoltar(Ef, Ep, Nf, Np)

    @staticmethod
    def calculate_wong1(Ef, Ep, Nf, Np):
        return calculate_wong1(Ef, Ep, Nf, Np)