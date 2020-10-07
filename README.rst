RankerEval
=========

.. image:: https://img.shields.io/pypi/v/rankereval.svg
    :target: https://pypi.python.org/pypi/rankereval
    :alt: Latest PyPI version

.. image:: https://github.com/microsoft/rankerEval/workflows/Python%20package/badge.svg
   :target: https://github.com/microsoft/rankerEval/actions
   :alt: Latest GitHub actions build status

Find the full `documentation here <https://rankereval.readthedocs.io>`_.

.. inclusion-marker-start


A fast numpy-based implementation of ranking metrics for information retrieval and recommendation.
Coded with efficiency in mind and support for edge cases. 

Features
--------
* Wide array of evaluation metrics for information retrieval and top-N recommender systems
* Flexible input formats: Supports masked arrays and sparse matrices 
* Built-in support for confidence intervals via bootstrapping
* Minimal dependencies: Numpy (required), SciPy (optional)
  
Usage
-----
.. code-block:: python

	from rankereval import BinaryLabels, Rankings, Recall
	
	y_true = BinaryLabels.from_positive_indices([[0,2], [0,1,2]])
	y_pred = Rankings.from_ranked_indices([[2,1], [1]])

	recall_at_3 = Recall(3).mean(y_true, y_pred)
	print(recall_at_3["score"])


To get confidence intervals (95% by default), specify ``conf_interval=True``:

.. code-block:: python

	recall_at_3 = Recall(3).mean(y_true, y_pred, conf_interval=True)
	print(recall_at_3["conf_interval"])
	
Input formats
+++++++++++++
RankerEval allows for a variety of input formats, e.g., 

.. code-block:: python

	# specify positive indices only
	y_true = BinaryLabels.from_dense([[1,0,1], [1,1,1]])
	
	# or use sparse matrix
	import scipy.sparse as sp
	y_true = BinaryLabels.from_sparse(sp.coo_matrix([[1,0,1], [1,1,1]]))


	
Installation
------------

To install (requires Numpy 1.18 or newer):

.. code-block:: bash

    pip install rankereval



Licence
-------
This project is licensed under `MIT <https://choosealicense.com/licenses/mit/>`_.

.. inclusion-marker-end

Authors
-------

`RankerEval` was written by `Tobias Schnabel <tobias.schnabel@microsoft.com>`_.


Contributing
------------

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_.
For more information see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or
contact `opencode@microsoft.com <mailto:opencode@microsoft.com>` with any additional questions or comments.

