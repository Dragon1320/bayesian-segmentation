from mpmath import mp, gamma

import numpy as np
import matplotlib.pyplot as plt

# parameters for generating an example sequence
n1 = 20
n2 = 20
p1 = 0.9
p2 = 0.1

# priors
pri_a = 1
pri_b = 1

# generate an example sequence where the first n1 elements have a different probability
# of heads/tails than the next n2 elements
def generate_example_seq(n1, n2, p1, p2):
  seq = np.concatenate([np.random.binomial(1, p1, n1), np.random.binomial(1, p2, n2)])

  return seq

# this calculation represents the inner portion of the loop
def infer_prob_seq(seq, pri_a, pri_b):
  n = len(seq)
  heads = (seq == 1).sum()
  tails = (seq == 0).sum()

  # ensure sequence only consists of 0s and 1s
  assert(n == heads + tails)

  p = ((gamma(pri_a + pri_b)) / (gamma(pri_a) * gamma(pri_b))) * ((gamma(heads + pri_a) * gamma(tails + pri_b)) / (gamma(n + pri_a + pri_b)))

  return p

# this calculation represents the loop and prior
def infer_prob_brk(seq, pos, pri_brk, pri_a, pri_b):
  n = len(seq)

  left = seq[:pos]
  right = seq[pos:]

  # since were doing a loop product, we initialise p = 1,
  # if this was a loop summation, we would instead initialise p = 0
  p = 1

  for sub_seq in [left, right]:
	  p *= infer_prob_seq(sub_seq, pri_a, pri_b)

  return p * pri_brk

# calculations
seq = generate_example_seq(n1, n2, p1, p2)
print("sequence: %s" % (seq))

# g(a) - breakpoint equally likely at each position in the sequence
pri_brk = 1 / (len(seq) + 1)

p_pos = []

for pos in range(len(seq)):
  p_brk = infer_prob_brk(seq, pos, pri_brk, pri_a, pri_b)

  p_pos.append(p_brk)

# calculate P(y_obs)
p_obs = sum(p_pos)

# plotting
scale_x = np.arange(0, len(seq))
scale_y = []

for x in scale_x:
  y = p_pos[x] / p_obs

  scale_y.append(y)

plt.bar(scale_x, scale_y)
plt.savefig("single_brk.png")
