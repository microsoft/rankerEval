============
Data Formats
============
rankereval distinguishes explicitly between ground truth labels and predictions. 
This avoids accidental swapping of arguments and allows for additional type and consistency checking.

.. autosummary::
	:nosignatures:

	rankereval.data.BinaryLabels
	rankereval.data.NumericLabels
	rankereval.data.Rankings
	

Ground truth labels
-------------------
.. autoclass:: rankereval.data.BinaryLabels
   :members:
   :inherited-members:

.. autoclass:: rankereval.data.NumericLabels
   :members:
   :inherited-members:
   
   
Predicted rankings
------------------
.. autoclass:: rankereval.data.Rankings
   :members:
   :inherited-members: