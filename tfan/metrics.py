import math

def fit_power(xs, ys):
    # y = a x^alpha (log-log least squares)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    sxx = sum(x*x for x in lx) - (sum(lx)**2)/n
    sxy = sum(x*y for x,y in zip(lx,ly)) - (sum(lx)*sum(ly))/n
    alpha = sxy / sxx
    a = math.exp((sum(ly) - alpha*sum(lx)) / n)
    return a, alpha

def percentile(vals, p):
    vals = sorted(vals)
    if not vals: return 0.0
    k = (len(vals)-1) * (p/100.0)
    f = int(k); c = min(f+1, len(vals)-1)
    return vals[f] + (vals[c]-vals[f])*(k-f)
