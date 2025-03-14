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

def next_difficulty(history, b, a, gain, limiter):
    if not history:
        return 1.0, 600.0

    vTimeDelta = [x[0] for x in history[:len(b)+1]]
    vTimeDelta = [x-y for x,y in zip(vTimeDelta[:-1], vTimeDelta[1:])]
    vTimeDelta.extend([600] * (len(b) - len(vTimeDelta)))

    vPredBuffer = [x[2] for x in history[:len(a)]]
    vPredBuffer.extend([600] * (len(a) - len(vPredBuffer)))

    dFilteredInterval = sum(np.array(vTimeDelta) * b) - sum(np.array(vPredBuffer) * a)
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

    return history[0][1] * dAdjustmentFactor, dFilteredInterval

def rawsim(start, end, nethash, b, a, interval=72, gain=0.18, limiter=2.0, func=None):
    blocks = []
    time = start
    cd = nethash(time)
    while time < end:
        nd, pred = next_difficulty(blocks[-max(len(b),len(a))-1:][::-1], b, a, gain, limiter)
        if blocks and not len(blocks)%interval:
            cd = nd
        nh = nethash(time)
        nt = func(nd, nh)
        blocks.append( (round(time), cd, pred, (nh + nethash(time+nt)) / 2, nt) )
        time += nt
    return np.array(blocks)

from random import expovariate
def simulate(*args, **argv):
    return rawsim(func=lambda nd,nh: expovariate(1.0 / (600.0 * nd / nh)), *args, **argv)
def impulse(*args, **argv):
    return rawsim(func=lambda nd,nh: 600.0 * nd / nh, *args, **argv)

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
    history = [x for x in sorted(history)]
    diff = []
    for idx in range(2, len(history)-1):
        offset = min(idx-1, window, len(history)-1-idx)
        interval = (history[idx + offset][0] -
                    history[idx - offset][0]) / (2.0 * offset + 1)
        diff.append((history[idx][0], history[idx][1]*600.0/interval))
    return hashintervals(diff)

from csv import reader
def history_from_csv(filename):
    with open(filename, 'r') as csvfile:
        return [(int(t),float(d)) for n,t,d in reader(csvfile)]

def utility_function(blocks):
    # Integrate the difference from perfection
    e = sum(np.square(blocks[:,3]-blocks[:,1])*blocks[:,4]) / (blocks[-1][0] - blocks[0][0])
    return np.sqrt(e)

def xfrange(x, y, step):
    while x < y:
        yield x
        x += step

if __name__ == '__main__':
    #frc = history_from_csv('data/frc.csv')
    #print(u"Freicoin historical error: %f" % utility_function([(t,d) for n,t,d in frc]))

    #btc = history_from_csv('data/btc.csv')
    #print(u"Bitcoin historical error: %f" % utility_function([(t,d) for n,t,d in btc]))

    steps = [(0*144*600,   1.0),
             (1*144*600, 100.0),
             (7*144*600,   1.0),
             (9*144*600,   1.0)]

    samples = steps
    nethash = hashintervals(steps)

    #          n  c    G          L
    params = [( 2, 0.5, 0.15625,   1.065),
              ( 3, 0.5, 0.1328125, 1.065),
              ( 4, 0.5, 0.1171875, 1.065),
              ( 5, 0.5, 0.1015625, 1.065),
              ( 6, 0.5, 0.09375,   1.065),
              ( 7, 0.5, 0.078125,  1.065),
              ( 8, 0.5, 0.0703125, 1.065),
              ( 9, 0.5, 0.0703125, 1.065),
              (10, 0.5, 0.0625,    1.065)]

    best = None
    for cfg in params:
        n = cfg[0]
        c = cfg[1]
        g = cfg[2]
        L = cfg[3]
        fn = 'out/bessel,n=%d,L=%f.csv'%(n,L)
        if os.path.exists(fn):
            continue
        record = u""
        innerbest = None
        for G in [1.0/256, 1.0/128, 1.0/92, 1.0/64, 1.0/48, 1.0/32, 1.0/16, 1.0/8, 1.0/4]:
            if True:
                try:
                    b, a = signal.bessel(n, c, 'low')
                    b /= a[0] # Normalize
                    a /= a[0]
                    a = a[1:]
                except:
                    continue
                for w in [1]:
                    res = []
                    for i in range(8):
                        blks = simulate(samples[0][0], samples[-1][0], nethash, b, a, interval=w, gain=G, limiter=L)
                        res.append( (utility_function(blks), len(blks)) )
                    res = np.array(res)
                    quality = (n,c,w,G,L, stats.tmean(res[:,0]), stats.sem(res[:,0]), (samples[-1][0]-samples[0][0])/stats.tmean(res[:,1]))
                    print(u"n=%d,c=%f,w=%d,G=%f,L=%f: %f +/- %f, %f" % quality)
                    if best is None or quality[5] < best[5]:
                        best = quality
                    if innerbest is None or quality[5] < innerbest[5]:
                        innerbest = quality
                    record += "%d,%f,%d,%f,%f,%f,%f,%f\n" % quality
        fp = open(fn, 'w')
        fp.write(record)
        strng = u"Best of n=%d is c=%f,w=%d,G=%f,L=%f: %f +/- %f, %f\n" % innerbest
        fp.write(strng)
        print(strng)
        fp.close()
    print(u"Best is n=%d,c=%f,w=%d,G=%f,L=%f: %f +/- %f, %f" % best)
    #for w in range(1, 145):
    #for G in xfrange(0.0025, 0.3500, 0.0025):
    #for L in xfrange(1.0005, 1.3500, 0.0005):

# End of File
