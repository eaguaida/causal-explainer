import sys
import tempfile
import types
import unittest

import matplotlib

matplotlib.use("Agg")

if "ipywidgets" not in sys.modules:
    ipywidgets = types.ModuleType("ipywidgets")

    class IntSlider:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def interact(*args, **kwargs):
        return None

    def fixed(value):
        return value

    ipywidgets.IntSlider = IntSlider
    ipywidgets.fixed = fixed
    ipywidgets.interact = interact
    sys.modules["ipywidgets"] = ipywidgets

import torch
from matplotlib import pyplot as plt
from PIL import Image

from causal_explainer.benchmark.evaluation import blur_image
from causal_explainer.sfl.relevance_score import RelevanceScore
from causal_explainer.visuals.saliency import SaliencyMapVisualizer


class RelevanceScoreTests(unittest.TestCase):
    def test_even_mutants_stay_in_pass_bucket_for_even_n(self):
        scorer = RelevanceScore(device=torch.device("cpu"))
        sampled_tensor = torch.zeros((4, 1, 2, 1), dtype=torch.float32)
        mask = torch.tensor(
            [
                [[[1.0], [0.0]]],
                [[[0.0], [1.0]]],
                [[[1.0], [1.0]]],
                [[[0.0], [0.0]]],
            ],
            dtype=torch.float32,
        )

        scorer.calculate_relevance_scores([1.0, 2.0, 3.0, 4.0], sampled_tensor, mask, 4)

        self.assertTrue(torch.equal(scorer.Ep.cpu(), torch.tensor([[[4.0], [3.0]]])))
        self.assertTrue(torch.equal(scorer.Ef.cpu(), torch.tensor([[[0.0], [2.0]]])))
        self.assertTrue(torch.equal(scorer.Np.cpu(), torch.tensor([[[0.0], [1.0]]])))
        self.assertTrue(torch.equal(scorer.Nf.cpu(), torch.tensor([[[6.0], [4.0]]])))

    def test_odd_n_uses_all_pass_and_fail_scores_without_shape_errors(self):
        scorer = RelevanceScore(device=torch.device("cpu"))
        sampled_tensor = torch.zeros((5, 1, 2, 1), dtype=torch.float32)
        mask = torch.tensor(
            [
                [[[1.0], [0.0]]],
                [[[0.0], [1.0]]],
                [[[1.0], [1.0]]],
                [[[0.0], [0.0]]],
                [[[1.0], [0.0]]],
            ],
            dtype=torch.float32,
        )

        scorer.calculate_relevance_scores([1.0, 2.0, 3.0, 4.0, 5.0], sampled_tensor, mask, 5)

        self.assertTrue(torch.equal(scorer.Ep.cpu(), torch.tensor([[[9.0], [3.0]]])))
        self.assertTrue(torch.equal(scorer.Ef.cpu(), torch.tensor([[[0.0], [2.0]]])))
        self.assertTrue(torch.equal(scorer.Np.cpu(), torch.tensor([[[0.0], [6.0]]])))
        self.assertTrue(torch.equal(scorer.Nf.cpu(), torch.tensor([[[6.0], [4.0]]])))


class BlurImageTests(unittest.TestCase):
    def test_blur_image_preserves_cpu_device_and_dtype(self):
        image = torch.rand((1, 3, 8, 8), dtype=torch.float32)

        blurred = blur_image(image)

        self.assertEqual(blurred.device, image.device)
        self.assertEqual(blurred.dtype, image.dtype)
        self.assertEqual(tuple(blurred.shape), tuple(image.shape))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_blur_image_runs_on_cuda_inputs(self):
        image = torch.rand((1, 3, 8, 8), device="cuda", dtype=torch.float32)

        blurred = blur_image(image)

        self.assertEqual(blurred.device.type, "cuda")
        self.assertEqual(blurred.dtype, image.dtype)
        self.assertEqual(tuple(blurred.shape), tuple(image.shape))


class SaliencyVisualizerTests(unittest.TestCase):
    @staticmethod
    def _make_dataset(height, width):
        dataset = []
        for row in range(height):
            for col in range(width):
                value = float(row * width + col)
                dataset.append(
                    {
                        "position": (row, col),
                        "Ep": value,
                        "Ef": value + 1.0,
                        "Np": value + 2.0,
                        "Nf": value + 3.0,
                        "ochiai": value / 10.0,
                        "tarantula": value / 10.0,
                        "zoltar": value / 10.0,
                        "wong1": value / 10.0,
                    }
                )
        return dataset

    def _make_visualizer(self):
        temp_dir = tempfile.TemporaryDirectory()
        image_path = f"{temp_dir.name}/source.png"
        Image.new("L", (12, 12), color=128).save(image_path)
        return temp_dir, SaliencyMapVisualizer(image_path)

    def test_select_scores_uses_cached_lowercased_key(self):
        temp_dir, visualizer = self._make_visualizer()
        try:
            scores = {
                "Ep": None,
                "Ef": None,
                "Np": None,
                "Nf": None,
                "ochiai": None,
                "tarantula": None,
                "zoltar": None,
                "wong1": None,
            }

            self.assertEqual(visualizer._select_scores(scores, "ALL"), list(scores.keys()))
            self.assertEqual(visualizer._select_scores(scores, "Ochiai"), ["ochiai"])
            self.assertEqual(
                visualizer._select_scores(scores, "unknown"),
                ["tarantula", "ochiai", "zoltar", "wong1"],
            )
        finally:
            temp_dir.cleanup()

    def test_build_score_grids_and_plot_use_dataset_shape(self):
        temp_dir, visualizer = self._make_visualizer()
        try:
            dataset = self._make_dataset(3, 5)

            scores = visualizer._build_score_grids(dataset)

            self.assertEqual(scores["Ep"].shape, (3, 5))
            self.assertEqual(scores["Ef"].shape, (3, 5))
            self.assertEqual(scores["Ep"][2, 4], 14.0)
            self.assertAlmostEqual(scores["ochiai"][2, 4], 1.0 - 1.4)

            figure = visualizer._plot_scores(scores, ["Ep"])
            try:
                image_shapes = [image.get_array().shape for image in figure.axes[0].images]
                self.assertEqual(image_shapes, [(3, 5), (3, 5)])
            finally:
                plt.close(figure)
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
