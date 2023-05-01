# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:55:06 2022

@author: Arpit Mittal
"""

from GoQlearnerPlayer import QLearner
from GoBoard import GO
import pickle

listOfsimulatedGames = []
with open("listOfsimulatedMiniMaxGames", "rb") as fp:   # Unpickling
    listOfsimulatedGames = pickle.load(fp)

