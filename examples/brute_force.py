from math import comb

from mpmath import mp, gamma

import numpy as np
import matplotlib.pyplot as plt

# for generating example data
n1 = 40
n2 = 40
p1 = 1
p2 = 0

# priors

# にゃ～
cats = {
  "_": 1,
  "*": 1,
}

def generate_example_seq(n1, n2, p1, p2):
  seq = np.concatenate([np.random.binomial(1, p1, n1), np.random.binomial(1, p2, n2)])
  seq = ["_" if c == 0 else "*" for c in seq]

  return seq

# the prior for P(K) - priors can be functions too!
def get_pri_num_brks(num_brks):
  p = 0.5 / (num_brks + 1)

  return p

# the prior for P(A | K)
def get_pri_segmentation(num_brks, seq_len):
  p = comb(seq_len, num_brks) ** -1

  return p

# probability of observing the given sequence
def infer_prob_seq(seq, cats):
  n = len(seq)

  pri_sum = 0
  pri_gamma_mul = 1

  counts = {}
  n_counts = 0

  for cat, pri in cats.items():
    count = seq.count(cat)

    counts[cat] = count
    n_counts += count

    pri_sum += pri
    pri_gamma_mul *= gamma(pri)

  # ensure that there are no symbols in the sequence that
  # are not listed and given a prior probability in cats
  assert(n == n_counts)

  p_mul = 1

  for cat, pri in cats.items():
    count = counts[cat]

    p_mul *= gamma(count + pri)

  p = (gamma(pri_sum) / pri_gamma_mul) * (p_mul / gamma(n + pri_sum))

  return p

# this calculates the probability of the sequence having k breakpoints
# we use a brute-force approach here which clearly is not scalable
def infer_prob_num_brks(seq, num_brks, cats):
  assert(num_brks >= 0)
  assert(num_brks <= 2)

  # this is our first prior
  # all the segmentations (ie. ways to divide up a sequence given k breakpoints) are assumed equally likely
  pri_segmentation = get_pri_segmentation(num_brks, len(seq) - 1)

  # hardcoded brute-force

  # this is a simplified version of the single breakpoint calculation (0 breakpoints)
  if num_brks == 0:
    p = infer_prob_seq(seq, cats)

    return p * pri_segmentation

  # this is exactly the same calculation as the single breakpoint
  if num_brks == 1:
    p = 0

    for pos in range(1, len(seq)):
      left = seq[:pos]
      right = seq[pos:]

      p += infer_prob_seq(left, cats) * infer_prob_seq(right, cats)

    return p * pri_segmentation

  # this is essentially an expanded version of the single breakpoint calculation
  if num_brks == 2:
    p = 0

    # as we can see, the number of possible segmentations of any given sequence increases exponentially
    # this will not work for very long sequences with a large number of breakpoints
    for first_brk in range(1, len(seq)):
      for second_brk in range(first_brk + 1, len(seq)):
        left = seq[:first_brk]
        mid = seq[first_brk:second_brk]
        right = seq[second_brk:]

        p += infer_prob_seq(left, cats) * infer_prob_seq(mid, cats) * infer_prob_seq(right, cats)

    return p * pri_segmentation

# calculations
seq = generate_example_seq(n1, n2, p1, p2)
print("sequence: %s" % ("".join(seq)))

p_brks = []

for num_brks in range(3):
  # this is our second prior
  # higher number of breakpoints are assumed to be less likely
  pri_num_brks = get_pri_num_brks(num_brks)

  p = infer_prob_num_brks(seq, num_brks, cats) * pri_num_brks

  p_brks.append(p)

p_obs = sum(p_brks)

# plotting
scale_x = np.arange(3)
scale_y = []

for x in scale_x:
  y = p_brks[x] / p_obs

  scale_y.append(y)

plt.title("inferring the number of breakpoints")
plt.xlabel("number of breakpoints")
plt.ylabel("probability")

plt.bar(scale_x, scale_y)
plt.savefig("brute_force.png")
