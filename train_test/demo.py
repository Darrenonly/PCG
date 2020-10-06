#
#
# def f1score(acc,se,sp):
#     x1 = (2*se - acc * se -1)/se
#     x2 = (acc - sp) / (1-sp)
#     TP = x2 / x1
#     pr = TP / (TP + 1)
#     y1 = pr * se
#     y2 = pr + se
#     f1 =  (2*y1 / y2)
#     return f1
#
#
# print(f1score(0.9674, 0.9485, 0.9342))
# print(f1score(0.9705, 0.9512, 0.9320))
# print(f1score(0.9328, 0.8621, 0.9516))
# print(f1score(0.9356, 0.8529, 0.9573))
# print(f1score(0.9770, 0.9820, 0.9222))

import time
import requests
import json

