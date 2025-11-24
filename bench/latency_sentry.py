import time, json, random
from tfan.telemetry import latency_percentiles

def bench(op, iters=200):
    times=[]
    for _ in range(iters):
        t0=time.time(); op(); times.append((time.time()-t0)*1000)
    return latency_percentiles(times)

if __name__=="__main__":
    # placeholder op; replace with true TLS/SSA, TTW, PGU calls in CI
    P = bench(lambda: time.sleep(random.random()*0.001))
    print(json.dumps(P))
