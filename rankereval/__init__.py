"""rankerEval - A fast numpy-based implementation of ranking metrics for information retrieval and recommendation."""

__version__ = '0.1.0'
__author__ = 'Tobias Schnabel <tobias.schnabel@microsoft.com>'
__all__ = [
    'BinaryLabels',
    'NumericLabels',
    'Rankings',
    'Precision',
    'Recall',
    'F1',
    'HitRate',
    'AP',
    'ReciprocalRank',
    'DCG',
    'NDCG',
    'MeanRanks']

from .data import BinaryLabels, NumericLabels, Rankings
from .metrics import Precision, Recall, F1, HitRate, AP, ReciprocalRank, DCG, NDCG, MeanRanks
