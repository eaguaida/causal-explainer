def run(self, confidence_scores, img, masks, N):
    self.calculate_all_scores(confidence_scores, img, masks, N)
    dataset = self.create_pixel_dataset(img.shape)
    result = dataset.copy()  # Create a copy of the dataset
    ochiai_array = self.ochiai_array.copy()
    tarantula_array = self.tarantula_array.copy()
    zoltar_array = self.zoltar_array.copy()
    wong1_array = self.wong1_array.copy()
    self.reset()
    return result, ochiai_array, tarantula_array, zoltar_array, wong1_array