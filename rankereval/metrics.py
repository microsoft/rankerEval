import numpy as np
from .data import BinaryLabels, Labels, Rankings


class Metric(object):
    def score(self, y_true, y_pred):
        r"""
        Individual scores for each ranking.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.
        ValueError
                if `n_bootstrap_samples`, `confidence` or `nan_handling` contain invalid values.
        """
        if not isinstance(y_true, Labels):
            raise TypeError("y_true must be of type Labels")
        if not isinstance(y_pred, Rankings):
            raise TypeError("y_pred must be of type Rankings")

        y_pred_labels = y_true.get_labels_for(y_pred)

        return self._score(y_true, y_pred_labels)

    @classmethod
    def _bootstrap_ci(cls, scores, n_bootstrap_samples, confidence):
        if not isinstance(n_bootstrap_samples,
                          int) or n_bootstrap_samples <= 1:
            raise ValueError("n_bootstrap_samples must be int > 1")
        elif not isinstance(confidence, float) or confidence <= 0.0 or confidence >= 1.0:
            raise ValueError("Confidence must be float and 0 < confidence < 1")

        if len(scores):
            resamples = np.random.choice(
                scores, (len(scores), n_bootstrap_samples), replace=True)
            bootstrap_means = resamples.mean(axis=0)

            # Compute "percentile bootstrap"
            alpha_2 = (1 - confidence) / 2.0
            lower_ci = np.quantile(bootstrap_means, alpha_2)
            upper_ci = np.quantile(bootstrap_means, 1.0 - alpha_2)
            return (lower_ci, upper_ci)
        else:
            return (float('nan'), float('nan'))

    def mean(self, y_true, y_pred, nan_handling='drop',
             conf_interval=False, n_bootstrap_samples=1000, confidence=0.95):
        r"""
        Mean score over all ranking after handling NaN values.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels, see also above.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated.
        nan_handling : {'propagate', 'drop', 'zerofill'}, optional
                `'propagate'` (default):
                        Return NaN if any value is NaN
                `'drop'` :
                        Ignore NaN values
                `'zerofill'` :
                        Replace NaN values with zero
        conv_interval : bool, optional
                If True, then return bootstrapped confidence intervals of mean,
                otherwise interval is None.
                Defaults to False.
        n_bootstrap_samples : int, optional
                Number of bootstrap samples to draw.
        confidence : float, optional
                Indicates width of confidence interval. Default is 0.95 (95%).
        Returns
        -------
        mean: dict
                Dictionary with ``mean["score"]`` for the mean score and ``mean["conf_interval"]``
                for the confidence interval tuple `(lower CI, upper CI)`.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.
        ValueError
                if `n_bootstrap_samples`, `confidence` or `nan_handling` contain invalid
                values.
        """

        scores = self.score(y_true, y_pred)
        if nan_handling == 'drop':
            scores = scores[~np.isnan(scores)]
        elif nan_handling == 'zerofill':
            scores = np.nan_to_num(scores)
        elif nan_handling == "propagate":
            if np.isnan(scores).sum():
                scores = []
        else:
            raise ValueError(
                'nan_handling must be "propagate", "drop" or "zerofill"')

        if conf_interval:
            ci = self._bootstrap_ci(scores, n_bootstrap_samples, confidence)
        else:
            ci = None

        if len(scores):
            mean = scores.mean()
        else:
            mean = float('nan')

        return {"score": mean, "conf_interval": ci}


class BinaryMetric(Metric):
    def __init__(self, k):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Cutoff k needs to be integer > 0")
        self._k = k

    def name(self):
        return self.__class__.__name__.replace("K", str(self._k))

    def score(self, y_true, y_pred):
        if not isinstance(y_true, BinaryLabels):
            raise TypeError("y_true must be of type BinaryLabels")
        return super().score(y_true, y_pred)


class Precision(BinaryMetric):
    """
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0.
    """

    def _precision(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        n_relevant = np.sum(y_pred_labels[:, :self._k] == 1, axis=-1).filled(0)

        scores = n_relevant.astype(float) / self._k
        # not defined if there are no relevant labels
        scores[n_pos == 0] = np.NaN
        return scores

    def _score(self, y_true, y_pred_labels):
        return self._precision(y_true, y_pred_labels)

    def score(self, y_true, y_pred):
        r"""
        Computes Precision@k [MN]_ of each ranking *y* in `y_pred` as

        .. math::

                \mathrm{Precision}@k(y) &= \frac{\sum_{i=1}^{k'} \mathrm{rel}(y_i)}{k},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`. Per definition above,
        :math:`\mathrm{Precision}@k(y) = 0`.

        2. There are no relevant items in `y_true`: :math:`\mathrm{Precision}@k(y) =` NaN.
        This marks invalid instances explicitly and is consistent with Recall.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, Precision
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1,2,3]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> Precision(3).score(y_true, y_pred)
        array([0.        , 0.66666667])

        """
        return super().score(y_true, y_pred)


class TruncatedPrecision(BinaryMetric):
    """
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0.
    """

    def _precision(self, y_true, y_pred_labels):
        n_relevant = np.sum(y_pred_labels[:, :self._k] == 1, axis=-1).filled(0)
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])

        items = np.broadcast_to(~y_pred_labels.mask, y_pred_labels.shape)
        n_items_in_y_pred = items.sum(axis=1).flatten()

        # not defined if there are no relevant labels
        scores = np.NaN * np.zeros_like(n_relevant, dtype=float)
        valid = (n_items_in_y_pred > 0) & (n_pos > 0)

        scores[valid] = n_relevant[valid].astype(float) / np.minimum(n_items_in_y_pred[valid], self._k)
        return scores

    def _score(self, y_true, y_pred_labels):
        return self._precision(y_true, y_pred_labels)

    def score(self, y_true, y_pred):
        r"""
        Computes the *truncated* Precision@k [MN]_ of each ranking *y* in `y_pred` as

        .. math::

                \mathrm{Precision}@k(y) &= \frac{\sum_{i=1}^{k'} \mathrm{rel}(y_i)}{k'},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*. The difference to Precision@k is the normalization by `k'`
        instead of `k`.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`. In this case,
        :math:`\mathrm{Precision}@k(y) = ` NaN.

        2. There are no relevant items in `y_true`: :math:`\mathrm{Precision}@k(y) =` NaN.
        This marks invalid instances explicitly and is consistent with Recall.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, Precision
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1,2,3]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> Precision(3).score(y_true, y_pred)
        array([0.        , 1.0])

        """
        return super().score(y_true, y_pred)


class Recall(BinaryMetric):
    """
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0.
    """

    def _recall(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        n_relevant = np.sum(y_pred_labels[:, :self._k] == 1, axis=-1).filled(0)

        scores = np.NaN * np.zeros_like(n_relevant, dtype=float)
        scores[n_pos > 0] = n_relevant[n_pos > 0].astype(
            float) / n_pos[n_pos > 0]
        return scores

    def _score(self, y_true, y_pred_labels):
        return self._recall(y_true, y_pred_labels)

    def score(self, y_true, y_pred):
        r"""
        Computes Recall@k [MN]_ as the fraction of relevant results in `y_true` that were
        in the top *k* results of `y_pred`.
        More formally, the recall of each ranking *y* in `y_pred` with
        labels `y_true` is defined as

        .. math::

                \mathrm{Recall}@k(y) &=
                \frac{\sum_{i=1}^{k'} \mathrm{rel}(y_i)}{\left\| y_{true} \right\| _1},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`. Per definition above,
        :math:`\mathrm{Recall}@k(y) = 0`.

        2. There are no relevant items in `y_true`: :math:`\mathrm{Recall}@k(y) =` NaN. This
        marks invalid instances explicitly.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, Recall
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> Recall(3).score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([0., 1.])

        """
        return super().score(y_true, y_pred)


class F1(Precision, Recall):
    def score(self, y_true, y_pred):
        r"""
        Computes F1 [MN]_ as harmonic mean of Precision@k and Recall@k.
        More formally, the F1 score of each ranking *y* in `y_pred` is defined as

        .. math::

                \mathrm{F1}@k(y) = \frac{2*\big(\mathrm{Precision}@k(y) *
                \mathrm{Recall}@k(y)\big)}{\mathrm{Precision}@k(y) + \mathrm{Recall}@k(y)}.


        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`. In this case,
        :math:`\mathrm{Recall}@k(y) = \mathrm{Precision}@k(y) = 0` .

        2. If :math:`\mathrm{Recall}@k(y) = \mathrm{Precision}@k(y) = 0`,
        we define :math:`\mathrm{F1}@k(y) = 0`.

        3. There are no relevant items in `y_true`: :math:`\mathrm{F1}@k(y) =` NaN.

        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, F1
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> F1(3).score(y_true, y_pred)  #doctest: +NORMALIZE_WHITESPACE
        array([0. , 0.5])

        """
        return super().score(y_true, y_pred)

    def _score(self, y_true, y_pred_labels):
        recall = self._recall(y_true, y_pred_labels)
        precision = self._precision(y_true, y_pred_labels)

        product = 2 * recall * precision
        sm = recall + precision

        # return 0 for geometric mean if both are zero
        scores = np.zeros_like(product, dtype=float)
        valid = np.nan_to_num(product) > 0
        invalid = np.isnan(product)

        scores[valid] = product[valid] / sm[valid]
        scores[invalid] = np.NaN

        return scores


class HitRate(Recall):
    """
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0.
    """

    def score(self, y_true, y_pred):
        r"""
        Computes HitRate@k [MN]_ as the whether a relevant item occurs
        in *k* results of `y_pred`.
        Differs from Recall@k in that `y_true` has to contain exactly
        one element per row.
        More formally, the HitRate of each ranking *y* in `y_pred` with labels
        `y_true` is defined as

        .. math::

                \mathrm{HitRate}@k(y) &= \sum_{i=1}^{k'} \mathrm{rel}(y_i),

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary and exactly one relevant item per row.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`.
        Per definition above, :math:`\mathrm{HitRate}@k(y) = 0`.

        2. There is not exactly one relevant item in
        `y_true`: :math:`\mathrm{HitRate}@k(y) =` NaN.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, HitRate
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> HitRate(3).score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([0., 1.])

        """
        return super().score(y_true, y_pred)

    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        scores = self._recall(y_true, y_pred_labels)
        scores[n_pos != 1] = np.NaN  # Not defined for no or multiple positives
        return scores


class ReciprocalRank(BinaryMetric):
    """
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0.
    """

    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, :self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        # It is 1/rank if document appears in top k, 0 otherwise
        scores = np.max(labels / ranks, axis=-1, initial=0.0)
        scores[n_pos == 0] = np.NaN  # Not defined for no multiple positives

        return scores

    def score(self, y_true, y_pred):
        r"""
        Computes ReciprocalRank@k [NC]_ as the rank where the first relevant item
        occurs in the top *k* results of `y_pred`.
        More formally, the ReciprocalRank of each ranking *y* in `y_pred` is defined as

        .. math::

                \mathrm{ReciprocalRank}@k(y) &= \max_{i=1,\ldots,k'} \frac{\mathrm{rel}(y_i)}{i},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty or first relevant item appears beyond rank *k*.
        Per definition above, :math:`\mathrm{ReciprocalRank}@k(y) = 0`.

        2. There is no relevant item in `y_true`: :math:`\mathrm{ReciprocalRank}@k(y) =` NaN.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, ReciprocalRank
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,0,1], [1,2]])
        >>> ReciprocalRank(3).score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([0.5, 1. ])

        """
        return super().score(y_true, y_pred)


class MeanRanks(BinaryMetric):
    """
            Used for evaluating permutations of `y_true`. Does not accept *k* as it
            scores permutations.
    """

    def __init__(self):
        pass

    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels.filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        scores = np.sum(ranks * labels, axis=-1)
        scores[n_pos > 0] = scores[n_pos > 0] / n_pos[n_pos > 0]
        scores[n_pos == 0] = np.NaN
        return scores

    def score(self, y_true, y_pred):
        r"""
        Computes MeanRanks@k as the mean of ranks of relevant items `y_pred`.
        More formally, it is defined for each ranking *y* in `y_pred` as

        .. math::

                \mathrm{MeanRanks}(y) =
                \frac{\sum_{i=1}^{|y|} i\cdot\mathrm{rel}(y_i)}{\sum_{i=1}^{|y|} \mathrm{rel}(y_i)},

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at rank *i*
        in the ranking *y*.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. There is no relevant item in `y_true`: :math:`\mathrm{MeanRanks}(y) =` NaN.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, MeanRanks
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1]])
        >>> y_pred = Rankings.from_ranked_indices([[3,0,5], [1,2]])
        >>> MeanRanks().score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([2.5, 1. ])

        """
        return super().score(y_true, y_pred)


class AP(BinaryMetric):
    """
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0.
    """

    def _score(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, :self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        precision = np.cumsum(labels, axis=-1) / ranks

        scores = np.zeros_like(n_pos, dtype=float)
        scores[n_pos > 0] = np.sum(
            precision * labels, axis=-1) / np.clip(n_pos[n_pos > 0], None, self._k)
        scores[n_pos == 0] = np.NaN

        return scores

    def score(self, y_true, y_pred):
        r"""
        Computes AveragePrecision@k [MN]_, an approximation to the the area
        under the precision-recall curve, of each ranking *y* in `y_pred` as

        .. math::

                \mathrm{AveragePrecision}@k(y) &= \frac{1}{Z} \sum_{i=1}^{k'}
                \mathrm{rel}(y_i)\cdot \mathrm{Precision}@i(y),

                k' &= \min(k,|y|),

                Z &= \min \big(k, \left\| y_{true} \right\| _1\big);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of the item at
        rank *i* in the ranking *y*.

        .. note::

                Sometimes the denominator *Z* is defined with respect to only the
                retrieved or recommended set of items `y_pred`. This is not desirable
                as AP could be artificially inflated, e.g., by returning only one
                relevant item at the top
                and then filling up the ranking with and irrelevant items.


        Parameters
        ----------
        y_true : :class:`~rankereval.data.BinaryLabels`
                Ground truth labels, must be binary.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`.
        Per definition above, :math:`\mathrm{AveragePrecision}@k(y) = 0`.

        2. There are no relevant items in `y_true`:
        :math:`\mathrm{AveragePrecision}@k(y) =` NaN to make it consistent with other metrics.

        3. There are no relevant items in `y_pred` up to *k*:
        Per definition above, :math:`\mathrm{AveragePrecision}@k(y) = 0`.


        Examples
        --------
        >>> from rankereval import BinaryLabels, Rankings, AP
        >>> # use separate labels for each ranking
        >>> y_true = BinaryLabels.from_positive_indices([[0, 5],[1,2,3]])
        >>> y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])
        >>> AP(3).score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([0. , 0.66666667])

        """
        return super().score(y_true, y_pred)


class DCG(Metric):
    r"""
            Parameters
            ----------
            k : int
                    specifies number of top results `k` of each ranking to be evaluated.

            relevance_scaling : str, ['binary' (default), 'power']
                    Determines are relevance labels are transformed:

                    `'identity'`: (default)
                            :math:`f(\mathrm{rel}(y_i)) = \mathrm{rel}(y_i)`
                    `'power'`:
                            :math:`f(\mathrm{rel}(y_i)) = 2^{\mathrm{rel}(y_i)} - 1`

            log_base : str, ['e' (default), '2']
                    Determines what log base is used in denominator.
                    The smaller this value, the heavier emphasis on top-ranked documents.

                    `'e'` (default):
                            Natural logarithm :math:`\ln`
                    `'2'`:
                            :math:`\log_2`

            Notes
            -----
            The original definition of (n)DCG [KJ]_ uses 'identity' for `relevance_scaling`,
            but leaves the choice of `log_base` open.

            Raises
            ------
            ValueError
                    if `k` is not integer > 0 or `relevance_scaling` or `log_base` are invalid.

    """
    SCALERS = {'identity': lambda x: x,
               'power': lambda x: np.power(x, 2) - 1}
    LOGS = {'2': lambda x: np.log2(x), 'e': lambda x: np.log(x)}

    def __init__(self, k=None, relevance_scaling='identity', log_base='e'):
        self._k = k
        if relevance_scaling not in self.SCALERS:
            raise ValueError(
                "Relevance scaling must be 'identity' or 'power'.")
        if log_base not in self.LOGS:
            raise ValueError("Log base needs to be 'e' or '2'.")
        self._rel_scale = self.SCALERS[relevance_scaling]
        self._log_fct = self.LOGS[log_base]

    def _dcg(self, y_true, y_pred_labels):
        n_pos = y_true.get_n_positives(y_pred_labels.shape[0])
        labels = y_pred_labels[:, :self._k].filled(0)
        ranks = np.arange(1, labels.shape[1] + 1, dtype=float).reshape(1, -1)

        scores = np.sum(
            self._rel_scale(labels) /
            self._log_fct(
                ranks +
                1),
            axis=-
            1)
        scores[n_pos == 0] = np.NaN
        return scores

    def _score(self, y_true, y_pred_labels):
        return self._dcg(y_true, y_pred_labels)

    def score(self, y_true, y_pred):
        r"""
        Computes Discounted Cumulative Gain at k (DCG@k) [KJ]_ as a
        weighted sum of relevance labels of top *k* results of `y_pred`.
        More formally, the recall of each ranking *y* in `y_pred`
        with labels `y_true` is defined as

        .. math::

                \mathrm{DCG}@k(y) &= \sum_{i=1}^{k'} \frac{f(\mathrm{rel}(y_i))}{\log_b(i + 1)},

                k' &= \min(k,|y|);

        where :math:`\mathrm{rel}(y_i)` is the relevance label of
        the item at rank *i* in the ranking *y*.
        *f* is the `relevance_scaling` function and *b* the `log_base`
        parameters defined earlier.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels, binary or numeric.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. Ranking to be evaluated is empty, i.e., :math:`|y|=0`.
        Per definition above, :math:`\mathrm{DCG}@k(y) = 0`.

        2. There are no items with relevance > 0 in `y_true`:
        :math:`\mathrm{DCG}@k(y) =` NaN to make it consistent with other metrics.

        Examples
        --------
        >>> from rankereval import NumericLabels, Rankings, DCG
        >>> # use separate labels for each ranking
        >>> y_true = NumericLabels.from_dense([[1, 2, 3], [4, 5]])
        >>> y_pred = Rankings.from_ranked_indices([[0,2,1], [1,0]])
        >>> DCG(3).score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([ 5.61610776, 10.85443211])

        """
        return super().score(y_true, y_pred)


class NDCG(DCG):
    """
    For a description of the arguments, see :class:`DCG`.
    """

    def _score(self, y_true, y_pred_labels):
        dcg = self._dcg(y_true, y_pred_labels)
        ideal_labels = y_true.get_labels_for(y_true.as_rankings())
        idcg = self._dcg(y_true, ideal_labels)

        return dcg / idcg

    def score(self, y_true, y_pred):
        r"""
        Computes the *normalized* Discounted Cumulative Gain at k (nDCG@k) [KJ]_
        as a weighted sum of relevance labels of top *k* results of `y_pred`,
        normalized to the range [0, 1].
        More formally, the nDCG of each ranking *y* in `y_pred` with
        labels `y_true` is defined as

        .. math::

                \mathrm{nDCG}@k(y)  = \begin{cases} \frac{\mathrm{DCG}@k(y)}
                {\mathrm{IDCG}@k(y)} &\mbox{if } \mathrm{IDCG}@k(y) > 0 \\
                0 & \mbox{otherwise } \end{cases},

        where :math:`\mathrm{IDCG}@k(y)` is the maximum DCG@k value that can
        be achieved on *all* relevance labels (i.e., DCG@k of the sorted relevance labels).

        .. note::

                Sometimes IDCG is defined with respect to only
                the retrieved or recommended set of items *y*. This is not desirable
                as nDCG could be artificially inflated, e.g., by
                returning only one relevant item at the top
                and then filling up the ranking with and irrelevant items.

        Parameters
        ----------
        y_true : :class:`~rankereval.data.Labels`
                Ground truth labels, binary or numeric.
        y_pred : :class:`~rankereval.data.Rankings`
                Rankings to be evaluated. If `y_true` only contains one row,
                the labels in this row will be used for every ranking in `y_pred`.
                Otherwise, each row *i* in `y_pred` uses label row *i* in `y_true`.

        Returns
        -------
        computed_metric: ndarray, shape (n_rankings, )
                Computed metric for each ranking.

        Raises
        ------
        TypeError
                if `y_true` or `y_pred` are of incorrect type.

        Notes
        -----
        Edge cases:

        1. `y` is empty, i.e., :math:`|y|=0`. Per definition above, :math:`\mathrm{DCG}@k(y) = 0`.

        2. There are no items with relevance > 0 in `y_true`: :math:`\mathrm{DCG}@k(y) =` NaN
        to make it consistent with other metrics.

        3. There are no items with relevance > 0 in `y` up to *k*:  :math:`\mathrm{nDCG}@k(y) = 0`.

        Examples
        --------
        >>> from rankereval import NumericLabels, Rankings, NDCG
        >>> # use separate labels for each ranking
        >>> y_true = NumericLabels.from_dense([[1, 2, 3], [4, 5]])
        >>> y_pred = Rankings.from_ranked_indices([[0,2,1], [1,0]])
        >>> NDCG(3).score(y_true, y_pred) #doctest: +NORMALIZE_WHITESPACE
        array([0.81749351, 1.        ])

        """
        return super().score(y_true, y_pred)
