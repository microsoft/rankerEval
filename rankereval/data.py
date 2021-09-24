import numpy as np
import numpy.ma as ma
import warnings
import numba

try:
    import scipy.sparse as sp

    SCIPY_INSTALLED = True
except ImportError:
    SCIPY_INSTALLED = False


class InvalidValuesWarning(UserWarning):
    pass


def _pad_list(x, padding_val=-999):
    if not isinstance(x, list):
        raise TypeError("Expect list as argument")

    if len(x) == 0:
        x = [x]
    elif isinstance(x[0], (int, float)):
        x = [x]

    converted = list()
    for el in x:
        if isinstance(el, list):
            converted.append(el)
        elif isinstance(el, np.ndarray) and el.ndim == 1:
            converted.append(el.tolist())
        elif isinstance(el, np.ndarray) and el.ndim == 0:
            converted.append([el.tolist()])
        else:
            raise ValueError(
                "Input contained invalid list element type; expect list or 1D np.array"
            )
    x = converted

    max_len = max(map(len, x))
    padded_list = np.array([row + [padding_val] * (max_len - len(row)) for row in x])
    mask = np.array([[False] * len(row) + [True] * (max_len - len(row)) for row in x])
    return padded_list, mask


def _parse_numpy(y, allow_non_finite_numbers=False, allow_ma_array=False):
    if isinstance(y, list):
        x, mask = _pad_list(y)
        y = ma.masked_array(x, mask=mask)
    elif np.ma.isMaskedArray(y):
        if not allow_ma_array:
            raise ValueError(
                "Masked arrays not supported. Use 'valid_items' to mask out entries."
            )
    elif isinstance(y, np.ndarray):
        y = ma.masked_array(y)
    else:
        raise ValueError("Input arrays need to be either list or numpy array")

    if not np.issubdtype(y.dtype, np.number) or np.issubdtype(np.dtype, np.bool_):
        raise ValueError("Input must be numeric")

    y = np.atleast_2d(y)

    nonfinite_entries = ~np.isfinite(y)
    if np.any(nonfinite_entries):
        if allow_non_finite_numbers:
            warnings.warn(
                "Input contains NaN or Inf entries which will be ignored.",
                InvalidValuesWarning,
            )
            y.mask = y.mask | nonfinite_entries
        else:
            raise ValueError("Input contains NaN or Inf entries")
    if y.ndim != 2:
        raise ValueError("Input arrays need to be 1D or 2D.")

    return y


def _convert_to_int(x):
    if np.all(x == np.floor(x)):
        return x.astype(int)
    else:
        raise ValueError("Input indices cannot be float type")


class Labels(object):
    """
    Abstract class for ground truth labels.
    """

    pass


@numba.njit
def fast_lookup(A, B):
    """
    Numba accelerated version of lookup table
    """
    vals = np.zeros(B.shape, dtype=np.float32)
    if len(A) == len(B):
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if B[i, j] in A[i]:
                    vals[i, j] = A[i][B[i, j]]
    else:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if B[i, j] in A[0]:
                    vals[i, j] = A[0][B[i, j]]
    return vals


@numba.njit
def fast_length(A):
    vals = np.zeros(len(A), dtype=np.float32)
    for i in range(len(A)):
        vals[i] = len(A[i])
    return vals


class BinaryLabels(Labels):
    """
    Represents binary ground truth data (e.g., 1 indicating relevance).
    """

    def _from_indices(self, labels):
        labels_raw = _convert_to_int(_parse_numpy(labels))
        ltu = list()
        for row in labels_raw:
            d = numba.typed.Dict.empty(
                key_type=numba.types.int64, value_type=numba.types.float32
            )
            for col in row[row >= 0]:
                d[col] = numba.types.float32(1.0)
            ltu.append(d)
        self._labels = numba.typed.List(ltu)
        return self

    def get_labels_for(self, ranking):
        indices = ranking._indices
        m_labels = len(self._labels)
        m_indices = len(indices)

        if m_indices < m_labels:
            raise ValueError(
                "Gold labels contain %d rows, but input rankings only have %d rows"
                % (m_labels, m_indices)
            )
        else:
            retrieved = fast_lookup(self._labels, indices.filled(0))
        return ma.masked_array(retrieved, mask=indices.mask)

    def as_rankings(self):
        return Rankings.from_ranked_indices(self.to_list())

    @classmethod
    def from_sparse(cls, matrix):
        """
        Construct a binary labels instance from a :class:`scipy.sparse` matrix.

        Parameters
        ----------
        matrix : sparse matrix, one row per context (e.g., user or query)
                Specifies binary labels for each item

        Raises
        ------
        TypeError
                if `matrix` is not of type :class:`scipy.sparse`.
        ValueError
                if `matrix` is contains non-binary values.

        Examples
        --------
        >>> import scipy.sparse as sp
        >>> matrix = sp.coo_matrix([[0, 1], [1, 1]])
        >>> BinaryLabels.from_sparse(matrix) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.BinaryLabels...>
        """
        if not SCIPY_INSTALLED:
            raise ValueError("Please install scipy for sparse matrix support.")
        if not sp.issparse(matrix):
            raise TypeError("Matrix is not a sparse matrix.")
        unique_values = np.unique(sp.find(matrix)[2])
        if len(np.setdiff1d(unique_values, [0, 1])):
            raise ValueError("Matrix may only contain 0 and 1 entries.")
        return cls.from_positive_indices(matrix.tolil().rows.tolist())

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
        return BinaryLabels()._from_indices(indices)

    def get_n_positives(self, n_rankings):
        m_labels = len(self._labels)
        n_pos = fast_length(self._labels)
        if m_labels == 1:
            n_pos = np.tile(n_pos, n_rankings)
        return n_pos

    @classmethod
    def from_dense(cls, labels):
        """
        Construct a binary labels instance from dense data where each item's label is specified.

        Parameters
        ----------
        labels : array_like, one row per context (e.g., user or query)
                Contains binary labels for each item. Labels must either be in {-1, +1} or {0, 1}.
                                Must be 1D or 2D, but row lengths can differ.

        Raises
        ------
        ValueError
                if `labels` is of invalid shape, type or non-binary.

        Examples
        --------
        >>> BinaryLabels.from_dense([[0, 1, 1], [0, 0, 1]]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.BinaryLabels...>
        """

        labels = _parse_numpy(labels)
        if len(np.setdiff1d(np.unique(labels).compressed(), [0, 1])) and len(
            np.setdiff1d(np.unique(labels).compressed(), [-1, 1])
        ):
            raise ValueError("Labels need to be binary.")
        indices = list(map(lambda x: np.nonzero(x > 0)[0].tolist(), labels))
        return BinaryLabels()._from_indices(indices)

    def to_list(self):
        return list(map(lambda x: list(sorted(x.keys())), self._labels))

    def __str__(self):
        return str(self.to_list())


class NumericLabels(BinaryLabels):
    """
    Represents numeric ground truth data (e.g., relevance labels from 1-5).
    """

    @classmethod
    def from_dense(cls, labels):
        """
        Construct a numeric labels instance from dense data where each item's label is specified.

        Parameters
        ----------
        labels : array_like, one row per context (e.g., user or query)
                Contains numeric labels for each item. Must be 1D or 2D, but row lengths can differ.

        Raises
        ------
        ValueError
                if `labels` is of invalid shape or type.

        Examples
        --------
        >>> NumericLabels.from_dense([[0, 3, 4], [0, 2, 5]]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <rankereval.data.NumericLabels...>
        """
        labels = _parse_numpy(labels).astype(float)
        return cls()._set_labels(labels)

    def _set_labels(self, labels):
        self._labels_raw = labels
        ltu = list()
        for row in self._labels_raw:
            d = numba.typed.Dict.empty(
                key_type=numba.types.int64, value_type=numba.types.float32
            )
            for col in np.nonzero(~row.mask)[0]:
                if row[col] > 0:
                    d[col] = numba.types.float32(row[col])
            ltu.append(d)
        self._labels = numba.typed.List(ltu)
        return self

    def as_rankings(self):
        return Rankings.from_scores(self._labels_raw, allow_ma_array=True)

    def to_list(self):
        return list(map(lambda x: x.compressed().tolist(), self._labels_raw))

    def __str__(self):
        return str(self._labels_raw)


class Rankings(object):
    """
    Represents (predicted) rankings to be evaluated.
    """

    @classmethod
    def from_ranked_indices(cls, indices, valid_items=None):
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
        indices = _convert_to_int(_parse_numpy(indices))

        return cls()._set_indices(indices, valid_items)

    @classmethod
    def from_scores(cls, raw_scores, valid_items=None, allow_ma_array=False):
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
        scores = _parse_numpy(
            raw_scores, allow_non_finite_numbers=True, allow_ma_array=allow_ma_array
        )

        sorted_indices = (-scores).argsort(axis=-1, kind="stable")

        if isinstance(scores.mask, np.bool_):
            mask = scores.mask
        else:
            mask = np.take_along_axis(scores.mask, sorted_indices, axis=-1)

        indices = ma.masked_array(sorted_indices, mask=mask)

        return cls()._set_indices(indices, valid_items)

    def _set_indices(self, indices, valid_items):
        if valid_items is not None:
            valid_items = _convert_to_int(_parse_numpy(valid_items))
            if valid_items.shape[0] != indices.shape[0]:
                raise ValueError(
                    "Valid indices need have the same number of rows as raw_scores."
                )

            # Mask out all entries unless explicitly specified
            mask = np.full(indices.shape, True, dtype=bool)
            for i, valid_in_row in enumerate(valid_items):
                mask[i] = ~np.isin(indices[i].filled(-1), valid_in_row.compressed())

            indices.mask = mask
        self._indices = indices
        self._indices = _convert_to_int(_parse_numpy(self.to_list()))
        self._n_nonzeros = self._indices.count(axis=-1)
        return self

    def __str__(self):
        return str(self._indices)

    def to_list(self):
        return list(map(lambda x: x.compressed().tolist(), self._indices))
