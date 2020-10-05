============
Data Formats
============
RankEval distinguishes explicitly between ground truth labels and predictions. 
This avoids accidental swapping of arguments and allows for additional type and consistency checking.

.. autosummary::
	:nosignatures:

	rankeval.data.BinaryLabels
	rankeval.data.NumericLabels
	rankeval.data.Rankings
	

Ground truth labels
-------------------
.. autoclass:: rankeval.data.BinaryLabels
   :members:
   :inherited-members:

.. autoclass:: rankeval.data.NumericLabels
   :members:
   :inherited-members:
   
   
Predicted rankings
------------------
.. autoclass:: rankeval.data.Rankings
   :members:
   :inherited-members: