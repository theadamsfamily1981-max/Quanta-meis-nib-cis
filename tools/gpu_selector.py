#!/usr/bin/env python3
import sys
try:
    import torch
except Exception as e:
    print("Torch not installed:", e); sys.exit(1)
if not torch.cuda.is_available():
    print("CUDA not available."); sys.exit(0)
scores = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    scores.append((i, int(p.total_memory), p.major*10 + p.minor, p.multi_processor_count, p.name))
scores.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
order = [str(s[0]) for s in scores]
names = [s[4] for s in scores]
print("Recommended CUDA_VISIBLE_DEVICES=" + ",".join(order))
print("Device order:", names)
