#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:20:55 2020

@author: blakehillier
"""

from pandas import read_csv
import torch
from XLNetFed import CalcSentiment, TextPrep, Train

data = CalcSentiment(read_csv('FedTextData.csv', names=['Date','Text']), read_csv('UNH.csv'))
inpt, attMsk = TextPrep(data)
model = Train(inpt, attMsk, list(data.Econ_Perf))
torch.save(model, 'XLNetStock.pt')