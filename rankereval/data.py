import numpy as np
import warnings
import numba
import scipy.sparse as sp
import itertools
import numpy.ma as ma
import time


class Timer:
    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        interval = end - self._start
        print(f"{self._name} took {interval:.2f} s")


class InvalidValuesWarning(UserWarning):
    pass


class Labels(object):
    """
    Abstract class for ground truth labels.
    """
    pass


@numba.njit(parallel=True)
def fast_lookup(A_indptr, A_cols, A_data, B):
    """
    Numba accelerated version of lookup table
    """
    # Non-existing indices are assigned label of 0.0
    vals = np.zeros(B.shape, dtype=np.float32)

    n_rows_a = len(A_indptr) - 1
    if n_rows_a == len(B):
        for i in numba.prange(B.shape[0]):
            ind_start, ind_end = A_indptr[i], A_indptr[i+1]
            left_idx = np.searchsorted(A_cols[ind_start:ind_end], B[i])
            right_idx = np.searchsorted(A_cols[ind_start:ind_end], B[i], side='right')
            found = (left_idx != right_idx)
            vals[i][found] = A_data[ind_start:ind_end][left_idx[found]]
    else:
        for i in numba.prange(B.shape[0]):
            left_idx = np.searchsorted(A_cols, B[i])
            right_idx = np.searchsorted(A_cols, B[i], side='right')
            found = (left_idx != right_idx)
            vals[i][found] = A_data[left_idx[found]]
    return vals


class BinaryLabels(Labels):
    """
    Represents binary ground truth data (e.g., 1 indicating relevance).
    """
    binary = True

    @classmethod
    def from_positive_indices(cls, indices):
        """
        Construct a binary labels instance from sparse data where only positive items are specified.

        Parameters
        ----------
        indices : array_like, one row per context (e.g., user or query)
                Specifies positive indices for each sample. Must be 1D or 2D, but row lengths can differ.

        Raises
        ------
        ValueError
                if `indices` is of invalid shape, type or contains duplicate, negative or non-integer indices.

        Examples
        --------
        >>> BinaryLabels.from_positive_indices([[1,2], [2]]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.BinaryLabels...>
        """
        sp_matrix = cls._check_values(SparseMatrix.from_nonzero_indices(indices).tocsr(), binary=True)
        return cls()._init(sp_matrix)

    @classmethod
    def from_matrix(cls, labels):
        """
        Construct a binary labels instance from dense or sparse matrix where each item's label is specified.

        Parameters
        ----------
        labels : 1D or 2D array, one row per context (e.g., user or query)
                Contains binary labels for each item. Labels must be in {0, 1}.

        Raises
        ------
        ValueError
                if `labels` is of invalid shape, type or non-binary.

        Examples
        --------
        >>> BinaryLabels.from_matrix([[0, 1, 1], [0, 0, 1]]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.BinaryLabels...>
        """
        sp_matrix = cls._check_values(SparseMatrix.from_matrix(labels).tocsr(), binary=cls.binary)
        return cls()._init(sp_matrix)

    def get_labels_for(self, ranking, k=None):
        n_label_rows = self._labels.shape[0]
        n_ranking_rows = len(ranking)

        if n_ranking_rows < n_label_rows:
            raise ValueError(
                f"Gold labels contain {n_label_rows} rows, but input rankings only have {n_ranking_rows} rows")

        indices, mask = ranking.get_top_k(k)
        retrieved = fast_lookup(self._labels.indptr,
                                self._labels.indices,
                                self._labels.data,
                                indices)
        return ma.masked_array(retrieved, mask=mask)

    @staticmethod
    @numba.njit
    def _numba_is_binary(data):
        for v in data:
            if v != 0 and v != 1:
                return False
        return True

    def as_rankings(self):
        return Rankings.from_scores(self._labels.tocsr(copy=True), warn_empty=False)

    def _init(self, labels):
        self._labels = labels
        return self

    @classmethod
    def _check_values(cls, matrix, binary=True):
        if binary and not cls._numba_is_binary(matrix.data):
            raise ValueError("Matrix may only contain 0 and 1 entries.")
        nonfinite_entries = ~np.isfinite(matrix.data)
        if np.any(nonfinite_entries):
            raise ValueError("Input contains NaN or Inf entries")
        return matrix

    def labels_to_list(self):
        return self._labels.tolil().data.tolist()

    def indices_to_list(self):
        return self._labels.tolil().rows.tolist()

    def get_n_positives(self, n_rankings):
        n_label_rows = self._labels.shape[0]
        n_pos = self._labels.getnnz(axis=1)
        if n_label_rows == 1:
            n_pos = np.tile(n_pos, n_rankings)
        return n_pos

    def __str__(self):
        return str(self.indices_to_list())


class NumericLabels(BinaryLabels):
    """
    Represents numeric ground truth data (e.g., relevance labels from 1-5).
    """
    binary = False


class SparseMatrix(object):
    def __init__(self, idx_ptr, col_idx, data, shape=None):
        self.idx_ptr = idx_ptr
        self.col_idx = col_idx
        self.data = data
        if shape:
            self.shape = shape
        else:
            if len(col_idx):
                M = col_idx.max()
            else:
                M = 0
            self.shape = (len(idx_ptr) - 1, M)
        self.csr = sp.csr_matrix((self.data, self.col_idx, self.idx_ptr), copy=False, shape=self.shape)

    @classmethod
    def from_values(cls, data, keep_zeros=False):
        if isinstance(data, list):
            if len(data) == 0 or np.ndim(data[0]) == 0:
                data = [data]
            idx = [list(range(len(r))) for r in data]
            return cls.from_lil(idx, data, keep_zeros=keep_zeros)
        else:
            return cls.from_matrix(data, keep_zeros=keep_zeros)

    @classmethod
    def from_nonzero_indices(cls, indices):
        if sp.issparse(indices):
            x = indices.tocsr()
            return cls(x.indptr, x.indices, x.data, x.shape)
        else:
            return cls.from_lil(indices)

    @classmethod
    def from_matrix(cls, matrix, keep_zeros=False):
        if np.ma.isMaskedArray(matrix):
            raise ValueError("Masked arrays not supported.")
        elif isinstance(matrix, np.ndarray) or isinstance(matrix, list):
            if isinstance(matrix, list):
                matrix = np.asarray(matrix, dtype=object).astype(np.float32)
            matrix = np.atleast_2d(matrix)
            if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.bool_):
                raise ValueError("Input must be numeric")
            elif matrix.ndim != 2:
                raise ValueError("Input arrays need to be 1D or 2D.")
            if keep_zeros:
                matrix += 1 - matrix[np.isfinite(matrix)].min()
            x = sp.csr_matrix(matrix)
            if not keep_zeros:
                x.eliminate_zeros()
        elif sp.issparse(matrix):
            x = matrix.tocsr()
        else:
            raise ValueError("Input type not supported.")
        return cls(x.indptr, x.indices, x.data, x.shape)

    @classmethod
    def from_lil(cls, rows, data=None, dtype=np.float32, keep_zeros=False):
        if not isinstance(rows, list) and not isinstance(rows, np.ndarray):
            raise ValueError("Invalid input type.")
        if len(rows) == 0 or np.ndim(rows[0]) == 0:
            rows = [rows]

        idx_ptr = np.asarray([0] + [len(x) for x in rows], dtype=int).cumsum()
        try:
            col_idx = np.fromiter(itertools.chain.from_iterable(rows), dtype=int, count=idx_ptr[-1])
            if data is None:
                data = np.ones_like(col_idx, dtype=dtype)
            else:
                data = np.fromiter(itertools.chain.from_iterable(data), dtype=dtype, count=idx_ptr[-1])
                if keep_zeros:
                    data += 1 - data[np.isfinite(data)].min()
        except TypeError:
            raise ValueError("Invalid values in input.")
        if len(data) != len(col_idx):
            raise ValueError("rows and data need to have same length")

        instance = cls(idx_ptr, col_idx, data)
        if not keep_zeros:
            instance.csr.eliminate_zeros()
        return instance

    def max_nnz_row_values(self):
        """Returns maximum number of non-zero entries in any row."""
        return (self.idx_ptr[1:] - self.idx_ptr[:-1]).max()

    def count_empty_rows(self):
        return ((self.idx_ptr[1:] - self.idx_ptr[:-1]) == 0).sum()

    def sort(self):
        self._numba_sort(self.idx_ptr, self.col_idx, self.data)

    def intersection(self, other):
        self._setop(other, True)

    def difference(self, other):
        self._setop(other, False)

    def isfinite(self):
        return np.all(np.isfinite(self.data))

    def remove_infinite(self):
        if not self.isfinite():
            self.data[~np.isfinite(self.data)] = 0
            self.csr.eliminate_zeros()

    def _setop(self, other, mode):
        if self.shape[0] != other.shape[0]:
            raise ValueError("Matrices need to have the same number of rows!")
        self._numba_setop(self.idx_ptr, self.col_idx, self.data, other.idx_ptr, other.col_idx, mode)
        self.csr.eliminate_zeros()

    def tocsr(self):
        return self.csr

    def tolil(self):
        res = []
        for i in range(len(self.idx_ptr) - 1):
            start, end = self.idx_ptr[i], self.idx_ptr[i+1]
            res += [self.col_idx[start:end].tolist()]
        return res

    def todense(self):
        pointers = (self.col_idx, self.col_idx, self.idx_ptr)
        return np.asarray(sp.csr_matrix(pointers, copy=False, shape=self.shape).todense())

    @staticmethod
    @numba.njit(parallel=True)
    def _numba_sort(idx_ptr, col_idx, data):
        for i in numba.prange(len(idx_ptr) - 1):
            start, end = idx_ptr[i], idx_ptr[i+1]
            args = (-data[start:end]).argsort(kind="mergesort")
            data[start:end] = data[start:end][args]
            col_idx[start:end] = col_idx[start:end][args]

    @staticmethod
    @numba.njit(parallel=True)
    def _numba_setop(self_idx_ptr, self_col_idx, self_data, other_idx_ptr, other_col_idx, intersect):
        for i in numba.prange(len(self_idx_ptr) - 1):
            ss, se = self_idx_ptr[i], self_idx_ptr[i+1]
            os, oe = other_idx_ptr[i], other_idx_ptr[i+1]

            left_idx = np.searchsorted(other_col_idx[os:oe], self_col_idx[ss:se])
            right_idx = np.searchsorted(other_col_idx[os:oe], self_col_idx[ss:se], side='right')
            if intersect:
                found = (left_idx == right_idx)
            else:
                found = (left_idx != right_idx)
            self_data[ss:se][found] = 0

    def __str__(self):
        return str((self.idx_ptr, self.col_idx, self.data))


class Rankings(object):
    """
    Represents (predicted) rankings to be evaluated.
    """

    def __init__(self, indices, valid_items=None, invalid_items=None, warn_empty=True):
        if valid_items is not None:
            valid_items = SparseMatrix.from_nonzero_indices(valid_items)
            indices.intersection(valid_items)
        if invalid_items is not None:
            invalid_items = SparseMatrix.from_nonzero_indices(invalid_items)
            indices.difference(invalid_items)
        if not indices.isfinite():
            warnings.warn("Input contains NaN or Inf entries which will be ignored.",
                          InvalidValuesWarning)
            indices.remove_infinite()
        n_empty_rows = indices.count_empty_rows()
        if n_empty_rows and warn_empty:
            warnings.warn(f"Input rankings have {n_empty_rows} empty rankings (rows). "
                          + "These will impact the mean scores." + str(indices.csr.todense()),
                          InvalidValuesWarning)
        self.indices = indices

    @classmethod
    def from_ranked_indices(cls, indices, valid_items=None, invalid_items=None):
        """
        Construct a rankings instance from data where item indices are specified in ranked order.

        Parameters
        ----------
        indices : array_like, one row per ranking
                Indices of items after ranking. Must be 1D or 2D, but row lengths can differ.
        valid_items : array_like, one row per ranking
                Indices of valid items (e.g., candidate set). Invalid items will be discarded from ranking.

        Raises
        ------
        ValueError
                if `indices` or `valid_items` of invalid shape or type.

        Examples
        --------
        >>> Rankings.from_ranked_indices([[5, 2], [4, 3, 1]]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.Rankings...>
        """
        indices = SparseMatrix.from_lil(indices)
        return cls(indices, valid_items, invalid_items)

    @classmethod
    def from_scores(cls, raw_scores, valid_items=None, invalid_items=None, warn_empty=True):
        """
        Construct a rankings instance from raw scores where each item's score is specified.
        Items will be ranked in descending order (higher scores meaning better).

        Parameters
        ----------
        raw_scores : array_like, one row per ranking
                Contains raw scores for each item. Must be 1D or 2D, but row lengths can differ.
        valid_items : array_like, one row per ranking
                Indices of valid items (e.g., candidate set). Invalid items will be discarded from ranking.

        Raises
        ------
        ValueError
                if `raw_scores` or `valid_items` of invalid shape or type.

        Warns
        ------
        InvalidValuesWarning
                if `raw_scores` contains non-finite values.

        Examples
        --------
        >>> Rankings.from_scores([[0.1, 0.5, 0.2], [0.4, 0.2, 0.5]]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.Rankings...>
        """
        indices = SparseMatrix.from_values(raw_scores, keep_zeros=True)
        indices.sort()

        return cls(indices, valid_items, invalid_items, warn_empty=warn_empty)

    def __str__(self):
        return str(self.indices)

    def __len__(self):
        return self.indices.shape[0]

    def to_list(self):
        return self.indices.tolil()

    def get_top_k(self, k=None):
        if k is None:
            k = self.indices.max_nnz_row_values()
        return self._csr_to_dense_masked(self.indices.idx_ptr, self.indices.col_idx, (len(self), k))

    @staticmethod
    @numba.njit
    def _csr_to_dense_masked(idx_ptr, col_idx, shape):
        condensed = np.zeros(shape, dtype=col_idx.dtype)
        mask = np.ones(shape, dtype=np.bool_)
        for i in range(len(idx_ptr) - 1):
            start, end = idx_ptr[i], idx_ptr[i+1]
            length = min(end - start, shape[1])
            condensed[i][:length] = col_idx[start:start+length]
            mask[i][:length] = False
        return condensed, mask


class TopKMixin:
    @staticmethod
    def topk(x, k, return_scores=False):
        # partition into k largest elements first
        index_array = np.sort(np.argpartition(-x, kth=k-1, axis=-1)[:, :k])
        top_k_partition = np.take_along_axis(x, index_array, axis=-1)

        # stable argsort in descending order
        top_idx_local = top_k_partition.shape[1] - 1
        top_idx_local -= np.fliplr(np.argsort(np.fliplr(top_k_partition), axis=-1, kind='stable'))

        # sort the top partition
        top_idx = np.take_along_axis(index_array, top_idx_local, axis=-1)
        if not return_scores:
            return top_idx
        else:
            top_scores = np.take_along_axis(top_k_partition, top_idx_local, axis=-1)
            return top_scores, top_idx


class DenseRankings(Rankings, TopKMixin):
    """
    Data structure where rankings have the same length (approximately).
    """

    def __init__(self, indices, mask=None, warn_empty=True):
        n_empty_rows = ((~mask).sum(axis=1) == 0).sum()
        if n_empty_rows and warn_empty:
            warnings.warn(f"Input rankings have {n_empty_rows} empty rankings (rows). "
                          + "These will impact the mean scores." + str(indices.csr.todense()),
                          InvalidValuesWarning)
        self.indices = indices
        self.mask = mask

    @classmethod
    def _verify_input(cls, arr, dtype=np.floating):
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input needs to be a numpy matrix")
        arr = np.asarray(np.atleast_2d(arr))
        if arr.ndim != 2:
            raise ValueError("Input arrays need to be 1D or 2D.")
        elif not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"Input array needs to be of type {dtype}")

        if np.issubdtype(dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                warnings.warn("Input contains NaN or Inf entries which will be ignored.",
                              InvalidValuesWarning)
                arr[~np.isfinite(arr)] = np.NINF
        elif not np.issubdtype(dtype, np.integer):
            raise TypeError("dtype argument must be floating or int")
        return arr

    @classmethod
    def from_ranked_indices(cls, indices, valid_items=None, invalid_items=None):
        """
        Set indices to -1 (or any other negative value) to indicate invalid index
        """
        indices = cls._verify_input(indices, dtype=np.integer)

        if valid_items is not None or invalid_items is not None:
            raise NotImplementedError("Not implemented yet")
        mask = (indices < 0)
        return cls(indices, mask)

    @classmethod
    def from_scores(cls, raw_scores, valid_items=None, invalid_items=None, warn_empty=True, k_max=None):
        raw_scores, mask = cls._verify_input(raw_scores, dtype=np.floating)

        if valid_items is not None:
            invalid_idx = SparseMatrix.from_nonzero_indices(invalid_items).csr.toarray() == 0
            raw_scores -= np.inf*invalid_idx
        if invalid_items is not None:
            invalid_items = SparseMatrix.from_nonzero_indices(invalid_items).csr
            raw_scores -= np.inf*invalid_items

        mask = ~np.isfinite(raw_scores)
        if k_max is None:
            sorted_idx = np.argsort(-raw_scores, axis=1, kind="stable")
        else:
            sorted_idx = cls.topk(raw_scores, k_max)
        mask = np.take_along_axis(mask, sorted_idx, axis=1)
        return cls(sorted_idx, mask)

    def get_top_k(self, k=None):
        if k is None:
            k = self.indices.shape[1]
        indices = self.indices[:, :k]
        mask = self.mask[:, :k]
        return indices, mask

    def to_list(self):
        return self.indices.tolist()
