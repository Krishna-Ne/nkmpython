# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:54:20 2020

@author: krishna
"""
import pandas
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import numpy

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)
#print("mean: % 3f" % x)

x = numpy.median(speed)
#print("median: % 3d" % x)

from scipy import stats
x = stats.mode(speed)
#print("mode"); print(x)

x = numpy.std(speed)
#print("Std deviation");print(x)

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31,73,97,13]

x = numpy.percentile(ages, 75)

#print("Percentilie");print(x)

#cGenerate a random normal distribution of size 2x3:
#loc - (Mean) where the peak of the bell exists.scale - (Standard Deviation) how flat the graph distribution should be.size - The shape of the returned array.
from numpy import random
import matplotlib.pyplot as plt

x = random.normal(loc=1, scale=4, size=(2, 3))
print("Normal distribution");print(x)
plt.hist(x, 10)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
#sns.distplot(random.normal(size=100), hist=False)
#plt.show()

# Binomial Distribution is a Discrete Distribution. ex binary scenarios toss of a coin
# it has three parameters:n - number of trials.
# p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).
# size - The shape of the returned array.
from numpy import random
x = random.binomial(n=10, p=0.5, size=10)
# print("binomial distribution");print(x)

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# sns.distplot(random.binomial(n=10, p=0.5, size=100), hist=True, kde=False)
# plt.show()

# Poisson Distribution is a Discrete Distribution.
# It estimates how many times an event can happen in a specified time. e.g. If someone eats twice a day what is probability he will eat thrice?
# It has two parameters: lam - rate or known number of occurences e.g. 2 for above problem.
#size - The shape of the returned array.
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

#csns.distplot(random.poisson(lam=2, size=1000), kde=False)
# plt.show()

# Chi Square distribution is used as a basis to verify the hypothesis.
# It has two parameters:df - (degree of freedom).
# size - The shape of the returned array.
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
# sns.distplot(random.chisquare(df=1, size=100), hist=False)
# cplt.show()

# Rayleigh distribution is used in signal processing.
# It has two parameters:
# scale - (standard deviation) decides how flat the distribution will be default 1.0).
# size - The shape of the returned array.
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
#sns.distplot(random.rayleigh(size=100), hist=False)
#plt.show()

# url = "https://raw.githubusercontent.com/callxpert/datasets/master/data-scientist-salaries.cc"
url = "C:/Users/krishna/Desktop/nk1252/ExecutivePoint/BITS/Data Analytics/Analytics with pythan training/salary experience data.csv"
salary = ['Years-experience', 'Salary']
dataset = pandas.read_csv(url, names=salary)
# shape
# print(dataset.shape)
#print(dataset.head(10))
# mean 
# print(dataset.describe())
from numpy import mean
from numpy import var
from numpy import std
#print('Mean: %.3f' % mean(dataset))
#print('varience: %.3f' % var(dataset))
#print('std deviation: %.3f' % std(dataset))
#print(dataset.describe())
