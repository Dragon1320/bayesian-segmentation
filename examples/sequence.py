# NOTE: we use mpmath because it deals with float overflows in a sane way (unlike numpy)
# realistically, if we throw large data at this, we will end up with values that are too
# large/small to be represented by regular python floats
from mpmath import mp, gamma

import numpy as np
import matplotlib.pyplot as plt

# parameters for generating an example sequence
n = 20
p = 0.7

# priors
# here we can define any prior assumptions/knowledge that we have that could influence our inference
# setting pri_a = pri_b = 1 in this case gives us a uniform distribution - an uninformative prior
pri_a = 1
pri_b = 1

# generate an example sequence of 1s and 0s where 1 represents heads and 0 represents tails
def generate_example_seq(n, p):
  seq = np.random.binomial(1, p, n)

  return seq

# calculate the probability of theta = x, given some sequence and priors
def infer_prob_seq(seq, x, pri_a, pri_b):
	# calculate values required for our algo based on the observed data
	n = len(seq)
	heads = (seq == 1).sum()
	tails = (seq == 0).sum()

  # ensure sequence only consists of 1s and 0s
	assert(n == heads + tails)

	# this is just subbing params into our formula
	# notice how we replace theta with x (we want to calculate P(theta = x))
	p = ((gamma(n + pri_a + pri_b)) / (gamma(heads + pri_a) * gamma(n - heads + pri_b))) * (x ** (heads + pri_a - 1)) * ((1 - x) ** (tails + pri_b - 1))

	return p

# calculations and plotting
seq = generate_example_seq(n, p)
print("sequence: %s" % (seq))

# since the PDF of P(theta) is continuous, we need to sample it at some interval
i = 0.05

scale_x = np.arange(0, 1 + i, i)
scale_y = []

# calculate the probability of theta = x, for all values of x between 0 and 1
for x in scale_x:
	p = infer_prob_seq(seq, x, pri_a, pri_b)

	scale_y.append(p)

plt.title("inferring the value of theta")
plt.xlabel("value of theta")
plt.ylabel("probability")

plt.plot(scale_x, scale_y)
plt.savefig("sequence.png")
