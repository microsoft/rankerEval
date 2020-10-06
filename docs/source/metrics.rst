============
Metrics
============
All metrics allow you to get both individual scores as well as mean scores:

.. autosummary:: 
   :nosignatures:

   rankereval.metrics.Metric.score
   rankereval.metrics.Metric.mean

The code below shows how ReciprocalRank@3 scores as well as *mean* ReciprocalRank@3 (MRR@3) can be computed:

.. code-block:: python

	from rankereval import BinaryLabels, Rankings, ReciprocalRank
	
	# define some input data
	y_true = BinaryLabels.from_positive_indices([[0, 5],[1,2,3]])
	y_pred = Rankings.from_ranked_indices([[3,2,1], [1,2]])

	# score rankings
	metric = ReciprocalRank(3)
	rr_scores = metric.score(y_true, y_pred)
	mrr = metric.mean(y_true, y_pred)

You can also obtain bootstrapped confidence intervals when passing `conf_interval=True`:

.. code-block:: python

	mrr = metric.mean(y_true, y_pred, conf_interval=True)
	print(mrr["conf_interval"])

.. note::

	rankereval was designed so that metrics return ``NaN`` values in ill-specified cases.
	If you observe a ``NaN`` value, please consult the documentation below and make sure you understand
	why it occurred. You can then either remove the ranking(s) from your input data, or handle them via the `nan_handling` option.
	
Base class
-----------
   
.. autoclass:: rankereval.metrics.Metric
   :members:

Precision@k
-----------
.. autoclass:: rankereval.metrics.Precision
   :members:
   
Recall@k
--------
.. autoclass:: rankereval.metrics.Recall
   :members:

F1@k
----
.. autoclass:: rankereval.metrics.F1
   :members:

HitRate@k
---------
.. autoclass:: rankereval.metrics.HitRate
   :members:

ReciprocalRank@k
----------------
.. autoclass:: rankereval.metrics.ReciprocalRank
   :members:

AveragePrecision@k
------------------
.. autoclass:: rankereval.metrics.AP
   :members:

MeanRanks
---------
.. autoclass:: rankereval.metrics.MeanRanks
   :members:

DCG@k
-----
.. autoclass:: rankereval.metrics.DCG
   :members:

NDCG@k
------
.. autoclass:: rankereval.metrics.NDCG
   :members:

References
----------
.. [MN] C. D. Manning, P. Raghavan and H. Schütze: 
		"Introduction to Information Retrieval". Cambridge University Press (2008).
.. [NC] N. Craswell: Mean Reciprocal Rank in: "Encyclopedia of Database Systems". Springer (2009).
.. [KJ] Kalervo Järvelin, Jaana Kekäläinen: "Cumulated gain-based evaluation of IR techniques". ACM Transactions on Information Systems 20(4), 422–446 (2002).