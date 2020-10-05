from rankereval import BinaryLabels, NumericLabels, Rankings
from rankereval import Precision, Recall, F1, HitRate, AP, ReciprocalRank, DCG, NDCG, MeanRanks

import pytest
from pytest import approx

BINARY_METRICS = [Precision, Recall, F1, HitRate, AP, ReciprocalRank]
y1 = [0, 5]
y2 = [8, 9]
y3 = []
y4 = [1, 2, 3, 4, 5, 6]
y5 = [3]
y6 = [0, 1]

r1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
r2 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
r3 = []
r4 = [1, 6, 8]

yn1 = [0, 3, 4]
yn2 = [2, 0, 5]

rn1 = [0, 1, 2]
rn2 = [2, 1, 0]


class TestBinaryMetricsInvalidInputs:
    @pytest.mark.parametrize("cls", BINARY_METRICS)
    def test_invalid_k(self, cls):
        with pytest.raises(ValueError):
            cls(None)

    @pytest.mark.parametrize("cls", BINARY_METRICS)
    def test_wrong_dim(self, cls):
        with pytest.raises(ValueError):
            cls(5).score(BinaryLabels.from_positive_indices(
                [[3], [4]]), Rankings.from_scores([[3]]))

    @pytest.mark.parametrize("cls", BINARY_METRICS)
    def test_invalid_numeric_input(self, cls):
        with pytest.raises(TypeError):
            cls(3).score(
                NumericLabels.from_dense(
                    [[]]), Rankings.from_scores(
                    [[]]))

    @pytest.mark.parametrize("cls", BINARY_METRICS)
    def test_invalid_ranking(self, cls):
        with pytest.raises(TypeError):
            cls(3).score(BinaryLabels.from_positive_indices([[]]), None)

    @pytest.mark.parametrize("cls", BINARY_METRICS)
    @pytest.mark.parametrize("params", [{"conf_interval": True, "n_bootstrap_samples": -1},
                                        {"conf_interval": True,
                                            "n_bootstrap_samples": 1},
                                        {"conf_interval": True,
                                            "n_bootstrap_samples": 3.5},
                                        {"conf_interval": True, "confidence": 0.0},
                                        {"conf_interval": True, "confidence": 1.0},
                                        {"conf_interval": True, "confidence": 3.5},
                                        {"conf_interval": True, "confidence": 3},
                                        {"nan_handling": "blub"}
                                        ])
    def test_bootstrap_invalid(self, cls, params):
        with pytest.raises(ValueError):
            cls(3).mean(BinaryLabels.from_positive_indices(y2),
                        Rankings.from_ranked_indices(r1), **params)


class TestRecall:
    @pytest.mark.parametrize("k,y_gold,y_pred,expect", [
        (3, y2, r1, [0.0]),
        (10, y2, r1, [1.0]),
        (10, y2, r2, [1.0]),
        (10, y2, r3, [0.0]),
        (10, y3, r3, [
            float('nan')]),
        (10, y3, r2, [
            float('nan')]),
        (9, y2, r1, [0.5]),
        (1, y4, r4, [1.0 / 6]),
        (1, y1, [r1, r2],
         [0.5, 0.0]),
        (1, [y1, y2], [
            r1, r2], [0.5, 0.5])
    ])
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = Recall(k).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestMean:
    @pytest.mark.parametrize("params,y_gold,y_pred,expect", [
        ({"nan_handling": "propagate"}, [y2, y2, y2, y3], [
         r1, r2, r3, r3], (float('nan'), None)),
        ({"nan_handling": "drop"}, [y2, y2, y2, y3], [
         r1, r2, r3, r3], (2.0 / 3, None)),
        ({"nan_handling": "zerofill"}, [y2, y2, y2, y3], [
         r1, r2, r3, r3], (2.0 / 4, None)),
        ({"nan_handling": "zerofill", "conf_interval": True},
         [y2, y2], [r1, r2], (1.0, (1.0, 1.0))),
        ({"nan_handling": "drop", "conf_interval": True}, [
         y3], [r3], (float('nan'), (float('nan'), float('nan'))))
    ])
    def test_mean(self, params, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = Recall(10).mean(y_gold, y_pred, **params)
        assert pred["score"] == approx(expect[0], nan_ok=True)
        if expect[1] is None:
            assert pred["conf_interval"] == expect[1]
        else:
            assert pred["conf_interval"] == approx(expect[1], nan_ok=True)

    @pytest.mark.parametrize("scores,n_bootstrap_samples,confidence,expect", [
        ([1.0, 1.0], 1000, 0.95, (1.0, 1.0)),
        ([0.5, 0.5], 1000, 0.95, (0.5, 0.5))
    ])
    def test_bootstrap_ci_exact(
            self, scores, n_bootstrap_samples, confidence, expect):
        pred = Recall._bootstrap_ci(scores, n_bootstrap_samples, confidence)
        assert pred == approx(expect, nan_ok=True)

    @pytest.mark.parametrize("scores,n_bootstrap_samples,confidence,expect", [
        ([0.0, 1.0], 1000, 0.9999999, (0.1, 0.9)),
        ([5.0, 10.0], 1000, 0.9999999, (6.0, 9.0))
    ])
    def test_bootstrap_ci_coverage(
            self, scores, n_bootstrap_samples, confidence, expect):
        pred = Recall._bootstrap_ci(scores, n_bootstrap_samples, confidence)
        assert pred[0] < expect[0]
        assert pred[1] > expect[1]

    @pytest.mark.parametrize("params", [
        ({"nan_handling": "other"}),
        ({"conf_interval": True, "n_bootstrap_samples": 1}),
        ({"conf_interval": True, "confidence": 0.0}),
        ({"conf_interval": True, "confidence": 1.0})
    ])
    def test_invalid(self, params):
        y_gold = BinaryLabels.from_positive_indices(y2)
        y_pred = Rankings.from_ranked_indices(y1)
        with pytest.raises(ValueError):
            Recall(10).mean(y_gold, y_pred, **params)


class TestPrecision:
    @pytest.mark.parametrize("k,y_gold,y_pred,expect", [
        (3, y2, r1, [0.0]),
        (10, y2, r1, [0.2]),
        (2, y2, r2, [1.0]),
        (10, y2, r3, [0.0]),
        (10, y3, r3, [
            float('nan')]),
        (10, y3, r2, [
            float('nan')]),
        (1, y2, r2, [1.0]),
        (3, y4, r4, [2.0 / 3]),
        (1, y1, [r1, r2],
         [1.0, 0.0]),
        (1, [y1, y2], [
            r1, r2], [1.0, 1.0])
    ])
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = Precision(k).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestF1:
    @pytest.mark.parametrize("k,y_gold,y_pred,expect", [
        (3, y2, r1, [0.0]),
        (10, y2, r1, [1.0 / 3]),
        (2, y2, r2, [1.0]),
        (10, y2, r3, [0.0]),
        (1, y4, r4, [0.285714286]),
        (10, y3, r3, [
            float('nan')]),
        (10, y3, r2, [
            float('nan')]),
        (1, y1, [r1, r2], [2.0 / 3, 0.0]),
        (1, [y1, y1], [r1, r2], [2.0 / 3, 0.0])
    ])
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = F1(k).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestAP:
    @pytest.mark.parametrize("k,y_gold,y_pred,expect", [
        (3, y2, r1, [0.0]),
        (10, y2, r1, [
            0.155555556]),
        (2, y2, r2, [1.0]),
        (10, y2, r3, [0.0]),
        (5, y4, r4, [2.0 / 5]),
        (10, y3, r3, [
            float('nan')]),
        (10, y3, r2, [
            float('nan')]),
        (6, y1, [r1, r2], [1.333333333 / 2, 0.2 / 2]),
        (6, [y5, y1], [r1, r2], [0.25, 0.2 / 2])
    ])
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = AP(k).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestHitrate:
    @pytest.mark.parametrize("k,y_gold,y_pred,expect", [
        (3, y5, r1, [0.0]),
        (3, y5, r2, [0.0]),
        (4, y5, r1, [1.0]),
        (4, y5, r2, [0.0]),
        (4, y1, r1, [
            float('nan')]),
        (4, y1, r2, [
            float('nan')]),
        (10, y3, r3, [
            float('nan')]),
        (10, y5, r3, [0.0]),
        (10, y3, r2, [
            float('nan')]),
        (5, y5, [r1, r2],
         [1.0, 0.0]),
        (5, [y5, y1], [r1, r2], [
            1.0, float('nan')])
    ])
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = HitRate(k).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestReciprocalRank:
    @pytest.mark.parametrize("k,y_gold,y_pred,expect", [
        (3, y5, r1, [0.0]),
        (3, y5, r2, [0.0]),
        (4, y5, r1, [1.0 / 4]),
        (4, y5, r2, [0.0]),
        (10, y1, r2, [1.0 / 5]),
        (10, y5, r2, [1.0 / 7]),
        (10, y3, r3, [
            float('nan')]),
        (10, y2, r3, [0.0]),
        (10, y3, r2, [
            float('nan')]),
        (5, y5, [r1, r2], [1.0 / 4, 0.0]),
        (5, [y5, y1], [r1, r2], [1.0 / 4, 1.0 / 5])
    ])
    def test_score(self, k, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = ReciprocalRank(k).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestMeanRanks:
    @pytest.mark.parametrize("y_gold,y_pred,expect", [
        (y1, r1, [7.0 / 2]),
        (y1, r2, [15.0 / 2]),
        (y3, r2, [float('nan')]),
        (y1, [r1, r2], [7.0 / 2, 15.0 / 2]),
        ([y1, y3], [r1, r2], [7.0 / 2, float('nan')])
    ])
    def test_score(self, y_gold, y_pred, expect):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = MeanRanks().score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestDCG:
    @pytest.mark.parametrize("y_gold,y_pred,expect,params", [
        (yn1, rn1, [5.616107762], {}),
        (yn1, rn2, [
            8.501497843], {}),
        (yn2, rn2, [6.0], {"log_base": "2"}),
        (yn1, r3, [0], {}),
        (y3, rn1, [
            float('nan')], {}),
        (y3, r3, [
            float('nan')], {}),
        (yn1, [rn1, rn2], [
            5.616107762, 8.501497843], {}),
        ([yn1, yn2], [rn1, rn2], [
            5.616107762, 8.656170245], {})
    ])
    def test_score(self, y_gold, y_pred, expect, params):
        y_gold = NumericLabels.from_dense(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = DCG(3, **params).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)


class TestNDCG:
    @pytest.mark.parametrize("y_gold,y_pred,expect,params", [
        (yn1, rn1, [5.616107762 / 8.501497843], {}),
        (yn2, rn1, [6.492127684 / 9.033953658], {}),
        (yn2, rn2, [8.656170245 / 9.033953658], {}),
        (yn1, r3, [0], {}),
        (y3, rn1, [
            float('nan')], {}),
        (y3, r3, [
            float('nan')], {}),
        (yn1, [rn1, rn2], [5.616107762 / 8.501497843, 1.0], {}),
        ([yn1, yn2], [rn1, rn2], [5.616107762 /
                                  8.501497843, 8.656170245 / 9.033953658], {})
    ])
    def test_numeric_score(self, y_gold, y_pred, expect, params):
        y_gold = NumericLabels.from_dense(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = NDCG(3, **params).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)

    @pytest.mark.parametrize("y_gold,y_pred,expect,params", [
        (y1, r1, [1.956593383 / 2.352934268], {}),
        (y6, rn1, [1.0], {}),
        (y1, r4, [0.0], {}),
        (y6, rn2, [1.631586747 / 2.352934268], {}),
        ([y1, y6], [r1, rn2], [1.956593383 / 2.352934268, 1.631586747 / 2.352934268], {})
    ])
    def test_binary_score(self, y_gold, y_pred, expect, params):
        y_gold = BinaryLabels.from_positive_indices(y_gold)
        y_pred = Rankings.from_ranked_indices(y_pred)
        pred = NDCG(10, **params).score(y_gold, y_pred).tolist()
        assert pred == approx(expect, nan_ok=True)
