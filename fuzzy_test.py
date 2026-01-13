import math
from collections import Counter
print(Counter( math.cos(42) for i in range(1000) ))