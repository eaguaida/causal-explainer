from sfl_techniques.formulas.fault_localization_metrics import FaultLocalizationMetrics

def calculate_all_scores(self, confidence_scores, img, masks, N):
    self.calculate_relevance_scores(confidence_scores, img, masks, N)
    ochiai_scores = FaultLocalizationMetrics.calculate_ochiai(self.Ef, self.Ep, self.Nf, self.Np)
    tarantula_scores = FaultLocalizationMetrics.calculate_tarantula(self.Ef, self.Ep, self.Nf, self.Np)
    zoltar_scores = FaultLocalizationMetrics.calculate_zoltar(self.Ef, self.Ep, self.Nf, self.Np)
    wong1_scores = FaultLocalizationMetrics.calculate_wong1(self.Ef, self.Ep, self.Nf, self.Np)

    self.ochiai_array = ochiai_scores.detach().cpu().numpy().max() - ochiai_scores.detach().cpu().numpy()
    self.tarantula_array = tarantula_scores.detach().cpu().numpy().max() - tarantula_scores.detach().cpu().numpy()
    self.zoltar_array = zoltar_scores.detach().cpu().numpy().max() - zoltar_scores.detach().cpu().numpy()
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