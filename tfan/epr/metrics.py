from math import sqrt

def coefficient_of_variation(series):
  values = [float(x) for x in series]
  if not values:
    return float('inf')
  mean = sum(values) / len(values)
  if mean == 0:
    return float('inf')
  variance = sum((v - mean) ** 2 for v in values) / len(values)
  return sqrt(variance) / abs(mean)
