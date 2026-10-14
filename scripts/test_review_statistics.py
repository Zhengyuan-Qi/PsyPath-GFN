import unittest
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error
from scripts.summarize_psypath_review_controls import scores_for_indices, normalize, METRICS


class ReviewStatisticsTests(unittest.TestCase):
    def test_legacy_prediction_reconstruction(self):
        result = dict(pids=["a", "b"], labels=[0, 1], threshold=.5,
                      probability={"posterior": [.2, .8]})
        rows = [dict(pid="b", label=1, learned_probability=[.7, .9], learned_severity=[8., 10.]),
                dict(pid="a", label=0, learned_probability=[.1, .3], learned_severity=[1., 3.])]
        got = normalize(result, rows)
        self.assertEqual(got["severity"], [2., 9.])
        rows[0]["learned_probability"] = [.2, .2]
        with self.assertRaises(AssertionError):
            normalize(result, rows)

    def test_resampled_metrics_match_direct_calculation(self):
        y = np.asarray([0, 1, 1, 0, 1, 0])
        q = np.asarray([0., 5., np.nan, 2., 7., 3.])
        p = np.asarray([.1, .6, .4, .6, .6, .2])
        s = np.asarray([1., 4., 3., 2., 6., 2.])
        ix = np.asarray([[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 4, 5], [3, 5, 1, 1, 4, 0]])
        result = scores_for_indices([dict(probability=p, severity=s, threshold=.5)], y, q, ix)[0]
        for i, indices in enumerate(ix):
            got = dict(zip(METRICS, result[i]))
            self.assertAlmostEqual(got["weighted_f1"], f1_score(y[indices], p[indices]>=.5, average="weighted"))
            self.assertAlmostEqual(got["positive_f1"], f1_score(y[indices], p[indices]>=.5))
            self.assertAlmostEqual(got["auroc"], roc_auc_score(y[indices], p[indices]))
            valid = np.isfinite(q[indices])
            self.assertAlmostEqual(got["mae"], mean_absolute_error(q[indices][valid], s[indices][valid]))


if __name__ == "__main__":
    unittest.main()
