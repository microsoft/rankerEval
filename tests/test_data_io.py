from rankereval import BinaryLabels, NumericLabels, Rankings
import pytest
import scipy.sparse as sp
import numpy as np

r1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
r2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
r3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


class TestBinaryLabelsDense:
    def test_multiple(self):
        pred = BinaryLabels.from_matrix([r1, r2, r3])
        gold = [[0, 5], [8, 9], []]
        assert pred.indices_to_list() == gold
        assert pred.get_n_positives(0).tolist() == [2, 2, 0]

    def test_one_dimensional(self):
        pred = BinaryLabels.from_matrix(r1).indices_to_list()
        gold = [[0, 5]]
        assert pred == gold

    def test_valid_empty(self):
        pred = BinaryLabels.from_matrix([]).indices_to_list()
        gold = [[]]
        assert pred == gold

    @pytest.mark.parametrize("invalid_value", [
        [[None]],
        [[float('nan')]],
        [["str"]],
        [[[0]]],
        [[1, 2]],
        None
    ])
    def test_invalid(self, invalid_value):
        with pytest.raises(ValueError):
            BinaryLabels.from_matrix(invalid_value)


class TestNumericLabelsDense:
    @pytest.mark.parametrize("input,expected", [
        ([[0, 5], [8, 9], [0, 0]], [[1], [0, 1], []]),
        ([[0, 4], [8, 9], [0, 0]], [[1], [0, 1], []]),
        ([[0, 5]], [[1]]),
        ([0, 5], [[1]]),
        ([[]], [[]])
    ])
    def test_io(self, input, expected):
        pred = NumericLabels.from_matrix(input).indices_to_list()
        assert pred == expected

    @pytest.mark.parametrize("input,expected", [
        ([[0, 5], [8, 9], [0, 0]], [[1], [0, 1], []]),
        ([[0, 4], [8, 9], [0, 0]], [[1], [0, 1], []]),
        ([[0, 5]], [[1]]),
        ([[0], [5]], [[], [0]]),
        ([[]], [[]])
    ])
    def test_np_io(self, input, expected):
        pred = NumericLabels.from_matrix([np.asarray(i) for i in input]).indices_to_list()
        assert pred == expected

    def test_counts(self):
        input = [[0, 5]]
        pred = NumericLabels.from_matrix(input).get_n_positives(1).tolist()
        assert pred == [1]

    def test_counts_broadcast(self):
        input = [[0, 5, 6]]
        pred = NumericLabels.from_matrix(input).get_n_positives(2).tolist()
        assert pred == [2, 2]

    @pytest.mark.parametrize("invalid_value", [
        [[None]],
        [[3], np.asarray([[0, 1]])],
        [[float('nan')]],
        [["str"]],
        [[[0]]],
        None
    ])
    def test_invalid(self, invalid_value):
        with pytest.raises(ValueError):
            NumericLabels.from_matrix(invalid_value)


class TestBinaryLabelsSparse:
    @pytest.mark.parametrize("input,expected", [
        ([r1, r2, r3], [[0, 5], [8, 9], []]),
        ([r1], [[0, 5]]),
        ([r3], [[]]),
        ([[]], [[]])
    ])
    def test_io(self, input, expected):
        matrix = sp.coo_matrix(input)
        pred = BinaryLabels.from_matrix(matrix).indices_to_list()
        assert pred == expected


class TestBinaryLabelsIndicies:
    @pytest.mark.parametrize("input,expected", [
        ([[0, 5], [8, 9], []], [[0, 5], [8, 9], []]),
        ([[4], [8, 9], []], [[4], [8, 9], []]),
        ([[0, 5]], [[0, 5]]),
        ([0, 5], [[0, 5]]),
        ([[]], [[]])
    ])
    def test_io(self, input, expected):
        pred = BinaryLabels.from_positive_indices(input).indices_to_list()
        assert pred == expected

    def test_counts_broadcast(self):
        input = [[0, 5, 6]]
        pred = BinaryLabels.from_positive_indices(
            input).get_n_positives(2).tolist()
        assert pred == [3, 3]

    @pytest.mark.parametrize("invalid_value", [
        [[None]],
        [[float('nan')]],
        [["str"]],
        [[[0]]],
        None
    ])
    def test_invalid(self, invalid_value):
        with pytest.raises(ValueError):
            BinaryLabels.from_positive_indices(invalid_value)


class TestRankings:
    @pytest.mark.parametrize("input,expected", [
        ([0, 1], [[0, 1]]),
        ([[4, 1], [2, 3]], [[4, 1], [2, 3]])
    ])
    def test_indices_io(self, input, expected):
        pred = Rankings.from_ranked_indices(input).to_list()
        print(type(pred))
        print(type(pred[0]))
        assert pred == expected

    @pytest.mark.parametrize("input,valid,expected", [
        ([0, 1], [1], [[1]]),
        ([[4, 1], [2, 3]], [[1, 4], [2]], [[4, 1], [2]])
    ])
    def test_indices_with_mask_io(self, input, valid, expected):
        pred = Rankings.from_ranked_indices(input, valid_items=valid).to_list()
        assert pred == expected

    @pytest.mark.parametrize("input,valid", [
        ([[4, 1], [2, 3]], [[0, 1]])
    ])
    def test_indices_with_invalid_mask(self, input, valid):
        with pytest.raises(ValueError):
            Rankings.from_ranked_indices(input, valid_items=valid).to_list()

    def test_invalid_mask_input(self):
        input = np.ma.masked_array([[0, 1, 2], [5, 4, 3]],
                                   mask=[[True, False, False], [False, True, False]])
        with pytest.raises(ValueError):
            Rankings.from_scores(input).to_list()

    @pytest.mark.parametrize("input", [
        [0, float('nan')],
        [[2, None]]
    ])
    def test_invalid_items(self, input):
        with pytest.raises(ValueError):
            Rankings.from_ranked_indices(input).to_list()

    @pytest.mark.parametrize("input,expected,has_warning", [
        ([0, 1], [[1, 0]], False),
        (np.asarray([0, 1]), [[1, 0]], False),
        (np.asarray([[0, 1], [3, 2]]), [[1, 0], [0, 1]], False),
        ([[4, 1], [2, 5, 6], [3, float('nan'), 4, float('inf')]],
         [[0, 1], [2, 1, 0], [2, 0]], True)
    ])
    def test_scores_io(self, input, expected, has_warning):
        if has_warning:
            with pytest.warns(UserWarning):
                pred = Rankings.from_scores(input).to_list()
        else:
            pred = Rankings.from_scores(input).to_list()
        print(expected, pred)
        assert pred == expected

    @pytest.mark.parametrize("input,valid,expected", [
        ([5, 0], np.asarray([0]), [[0]]),
        ([[1, 4], [3, 2]], [[0, 1], [1]], [[1, 0], [1]])
    ])
    def test_scores_io_with_valid_items(self, input, valid, expected):
        pred = Rankings.from_scores(input, valid_items=valid).to_list()
        assert pred == expected

    @pytest.mark.parametrize("input,invalid,expected, has_warning", [
        ([5, 0], [0], [[1]], False),
        ([[1, 4], [3, 2]], [[0, 1], [1]], [[], [0]], True)
    ])
    def test_scores_io_with_invalid_items(self, input, invalid, expected, has_warning):
        if has_warning:
            with pytest.warns(UserWarning):
                pred = Rankings.from_scores(input, invalid_items=invalid).to_list()
        else:
            pred = Rankings.from_scores(input, invalid_items=invalid).to_list()
        assert pred == expected
