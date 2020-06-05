#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np
from scipy import signal, stats

FILTER_COEFF = [
     -845859,  -459003,  -573589,  -703227,  -848199, -1008841,
    -1183669, -1372046, -1573247, -1787578, -2011503, -2243311,
    -2482346, -2723079, -2964681, -3202200, -3432186, -3650186,
    -3851924, -4032122, -4185340, -4306430, -4389146, -4427786,
    -4416716, -4349289, -4220031, -4022692, -3751740, -3401468,
    -2966915, -2443070, -1825548, -1110759,  -295281,   623307,
     1646668,  2775970,  4011152,  5351560,  6795424,  8340274,
     9982332, 11717130, 13539111, 15441640, 17417389, 19457954,
    21554056, 23695744, 25872220, 28072119, 30283431, 32493814,
    34690317, 36859911, 38989360, 41065293, 43074548, 45004087,
    46841170, 48573558, 50189545, 51678076, 53028839, 54232505,
    55280554, 56165609, 56881415, 57422788, 57785876, 57968085,
    57968084, 57785876, 57422788, 56881415, 56165609, 55280554,
    54232505, 53028839, 51678076, 50189545, 48573558, 46841170,
    45004087, 43074548, 41065293, 38989360, 36859911, 34690317,
    32493814, 30283431, 28072119, 25872220, 23695744, 21554057,
    19457953, 17417389, 15441640, 13539111, 11717130,  9982332,
     8340274,  6795424,  5351560,  4011152,  2775970,  1646668,
      623307,  -295281, -1110759, -1825548, -2443070, -2966915,
    -3401468, -3751740, -4022692, -4220031, -4349289, -4416715,
    -4427787, -4389146, -4306430, -4185340, -4032122, -3851924,
    -3650186, -3432186, -3202200, -2964681, -2723079, -2482346,
    -2243311, -2011503, -1787578, -1573247, -1372046, -1183669,
    -1008841,  -848199,  -703227,  -573589,  -459003,  -845858]

def next_difficulty(history, taps, gain, limiter):
    if not history:
        return 1.0

    vTimeDelta = [x[0] for x in history[:len(taps)+1]]
    vTimeDelta = [x-y for x,y in zip(vTimeDelta[:-1], vTimeDelta[1:])]
    vTimeDelta.extend([600] * (len(taps) - len(vTimeDelta)))

    dFilteredInterval = sum(np.array(vTimeDelta) * taps)
    tmp = (dFilteredInterval - 600.0) / 600.0
    if gain is not None:
        tmp *= gain
    dAdjustmentFactor = 1.0 - tmp

    if limiter is not None:
        max_limiter = limiter
        min_limiter = 1.0 / limiter
        if dAdjustmentFactor > max_limiter:
            dAdjustmentFactor = max_limiter
        elif dAdjustmentFactor < min_limiter:
            dAdjustmentFactor = min_limiter

    return history[0][1] * dAdjustmentFactor

from random import expovariate
def simulate(start, end, nethash, taps, interval=72, gain=0.18, limiter=2.0):
    blocks = []
    time = start
    nd = nethash(time)
    while time < end:
        if blocks and not len(blocks)%interval:
            nd = next_difficulty(blocks[-len(taps)-1:][::-1], taps, gain, limiter)
        nh = nethash(time)
        nt = expovariate(1.0 / (600.0 * nd / nh))
        blocks.append( (round(time), nd, (nh + nethash(time+nt)) / 2, nt) )
        time += nt
    return np.array(blocks)

from bisect import bisect_left
def hashintervals(diff):
    def nethash(time):
        if  time > diff[-1][0]:
            return diff[-1][1]
        return diff[max(0, bisect_left(diff, (time, 1.0))-1)][1]
    return nethash

def smooth(history, window=16):
    # Sort the history by time, so that we don't have any negative block
    # times. Not ideal, but allows us to avoid possible instability in the
    # simulator.
    history = [(int(n),int(t),float(d))
               for t,n,d in sorted((t,n,d) for n,t,d in history)]
    diff = []
    for idx in range(2, len(history)-1):
        offset = min(idx-1, window, len(history)-1-idx)
        interval = (history[idx + offset][1] -
                    history[idx - offset][1]) / (2.0 * offset + 1)
        diff.append((history[idx][1], history[idx][2]*600.0/interval))
    return hashintervals(diff)

from csv import reader
def history_from_csv(filename):
    with open(filename, 'r') as csvfile:
        return [(int(n),int(t),float(d)) for n,t,d in reader(csvfile)]

def utility_function(blocks):
    # Integrate the difference from perfection
    e = sum(np.square(blocks[:,2]-blocks[:,1])*blocks[:,3]) / (blocks[-1][0] - blocks[0][0])
    return np.sqrt(e)

def xfrange(x, y, step):
    while x < y:
        yield x
        x += step

if __name__ == '__main__':
    #frc = history_from_csv('data/frc.csv')
    #print(u"Freicoin historical error: %f" % utility_function([(t,d) for n,t,d in frc]))

    btc = history_from_csv('data/btc.csv')
    #print(u"Bitcoin historical error: %f" % utility_function([(t,d) for n,t,d in btc]))

    steps = [(0*144*600,   1.0),
             (1*144*600, 100.0),
             (7*144*600,   1.0),
             (9*144*600,   1.0)]
    nethash = hashintervals(steps)
    w = 9
    G = 0.125
    L = 1.375
    best = None
    for n in [2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,27,28,30,32,36,40,42,45,48,54,56,60,63,64,72,80,84,90,96,108,112,120,126,128,144]:
        fn = 'out/remez,n=%d.csv'%n
        if os.path.exists(fn):
            continue
        record = u""
        innerbest = None
        for c in xfrange(0.01, 0.49, 0.005):
            for cw in [0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
                if c - cw/2 < 0.001 or 0.499 < c + cw/2:
                    continue
                try:
                    taps = signal.remez(n, [0, c - cw/2, c + cw/2, 0.5], [1, 0], maxiter=50)
                    taps /= sum(taps)
                except:
                    continue
                innerbest = None
                for w in range(1,n+1):
                    res = []
                    for i in range(12):
                        blks = simulate(steps[0][0], steps[-1][0], nethash, taps, interval=w, gain=G, limiter=L)
                        res.append( (utility_function(blks), len(blks)) )
                    res = np.array(res)
                    quality = (n,c,cw,w,G,L, stats.tmean(res[:,0]), stats.sem(res[:,0]))
                    print(u"n=%d,c=%f,cw=%f,w=%d,G=%f,L=%f: %f +/- %f" % quality)
                    if best is None or quality[6] < best[6]:
                        best = quality
                    if innerbest is None or quality[6] < innerbest[6]:
                        innerbest = quality
                    record += "%d,%f,%f,%d,%f,%f,%f,%f\n" % quality
        fp = open(fn, 'w')
        fp.write(record)
        strng = u"Best of n=%d is c=%f,cw=%f,w=%d,G=%f,L=%f: %f +/- %f" % quality
        fp.write(strng)
        print(strng)
        fp.close()
    print(u"Best is n=%d,c=%f,cw=%f,w=%d,G=%f,L=%f: %f +/- %f" % best)
    #for w in range(1, 145):
    #for G in xfrange(0.0025, 0.3500, 0.0025):
    #for L in xfrange(1.0005, 1.3500, 0.0005):

# End of File
