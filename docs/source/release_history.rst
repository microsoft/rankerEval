===============
Release History
===============

Version 0.2.0 (2022-01-14)
----------------------------
* Rewrote underlying data structure used in Rankings for substantial speedup and memory efficiency
* Fixed shape error in AP
* Recall and Precision now have an optional truncated argument
* Default setting for NaN values is now to propagate them
* Smaller fixes and optimizations (forgotten print statements, pruning rankings early)

Version 0.1.2 (2021-06-24)
----------------------------
* Discontinued support for masked arrays as input because it can easily introduce mistakes
* Added valid_items parameter to Rankings to specify a candidate set

Initial Release (2020-10-10)
----------------------------