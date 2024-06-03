from math_utils import *
import time
import numpy as np

def zero_localizer(func, interval, d=1e-3):
    start = time.time()
    points = np.arange(interval[0], interval[1], d)
    target = func(points)
    for i in range(len(points)-1):
        if target[i]*target[i+1] < 0 and np.abs(d) > 1e-6:
            zero = zero_localizer(func, points[i:i+2], d=d*1e-1)
        elif target[i]*target[i+1] < 0 and np.abs(d) == 1e-6:
            zero, zero_time = find_zero(func, points[i:i+2], method='halley', time_this=True)
    end = time.time()
    print(f'Time elapsed: {round(end-start, 5)}s')
    print(f'Time elapsed while find_zero: {round(zero_time, 5)}s')