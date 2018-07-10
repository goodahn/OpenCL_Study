#!/usr/bin/python3
import sys
import numpy as np

result = []

with open(sys.argv[1]) as f:
	for line in f:
		if "mircro" in line:
			for word in line.split():
				try:
					result.append(int(word))
				except:
					continue

print(np.median(result))
