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