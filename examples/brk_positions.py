from math import comb

import numpy as np
from mpmath import mp, gamma

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

# we will only calculate breakpoint probabilities until k = 5
kmax = 5

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

# initialise the k = 0 base-case
def init_dp_array(seq, kmax, cats):
  # k starts at 0, account for this in the array size
  # we need the big boi floats for the calcs were doing owo
  dp = np.ndarray(shape=(kmax + 1, len(seq) + 1, len(seq) + 1), dtype=mp.mpf)

  # k = 0
  for j in range(0, len(seq)):
    for v in range(j, len(seq)):
      sub_seq = seq[j:v + 1]

      p =  infer_prob_seq(sub_seq, cats)

      # these values should be treated the same as seq[j:v]
      # namely, the element at idx v is not included
      dp[0][j][v + 1] = p

  return dp

# compose all the other calculations using our pre-calculated base-case
def infer_prob_num_brks(seq, kmax, cats, dp):
  # running this for anything else is dum
  assert(kmax >= 1)
  # this is important as we can only have as many breakpoints as n + 1 elements
  # ie. we cannot have 2 breakpoints in a sequence composed of 2 elements
  assert(kmax < len(seq))

  # now we use the stuff to calculate the other stuff
  # very descriptive ik, im somewhat of a poet

  # k = 1 -> k = kmax
  for k in range(1, kmax + 1):
    for j in range(0, len(seq) - k):
      p = 0

      for s in range(j + 1, len(seq) - (k - 1)):
        left = dp[0][j][s]
        right = dp[k - 1][s][len(seq)]

        p += left * right

      dp[k][j][len(seq)] = p

# use all the dynamic programming calculations to get final probabilities for all breakpoint counts >= kmax
def infer_prob_all_brks(seq, kmax, cats, dp):
  p_brks = []

  for k in range(0, kmax + 1):
    # apply prior P(A | K)
    p = dp[k][0][len(seq)] * get_pri_segmentation(k, len(seq) - 1)

    # apply prior P(K)
    pk = get_pri_num_brks(k)

    p_brks.append(p * pk)

  # calculate and apply marginal probability
  pr = sum(p_brks)
  p_brks = [p / pr for p in p_brks]

  return pr, p_brks

# calculate the positions of breakpoints given the number of breakpoints in the sequence
# NOTE: num_brks is different from kmax, it should be set to the most likely number
# of breakpoints given the sequence (see infer_prob_all_brks)
def get_brk_locations(seq, num_brks, cats, dp):
  # this tracks the location of the last breakpoint we found
  m = 0
  # this stores all the breakpoint locations we have identified
  brk_locs = []

  # we loop through all values from kmax -> 1 inclusive
  # this allows us to decrement k as described above once weve identified a breakpoint
  for q in reversed(range(1, num_brks + 1)):
    # store all results from the calculation: P(sequence | q) = P(left | 0)P(right | q - 1)
    # we can then find the location at which this result is maximised - our breakpoint location
    p_seg = []

    # m should never >= len(seq)
    seq_len = len(seq) - m

    # step through each position in our (sub) sequence
    # we cannot have q breakpoints in a sequence composed of <= q elements
    for j in range(m + 1, len(seq) - q):
      seq_len_right = len(seq) - j
      pri_segmentation_right = get_pri_segmentation(q - 1, seq_len_right - 1)

      # pri_segmentation for left will always be 1 (x choose 0 == 1)
      left = dp[0][m][j]
      right = dp[q - 1][j][len(seq)] * pri_segmentation_right

      p = left * right

      # we dont care about P(y_obs) - its just used for normalising our probabilities
      p_seg.append(p)

    # move m to the next breakpoint weve found
    m = m + np.argmax(p_seg)
    brk_locs.append(m)

  return brk_locs

# calculations
seq = generate_example_seq(n1, n2, p1, p2)
print("sequence: %s" % ("".join(seq)))

dp = init_dp_array(seq, kmax, cats)

infer_prob_num_brks(seq, kmax, cats, dp)

_, p_brks = infer_prob_all_brks(seq, kmax, cats, dp)
num_brks = np.argmax(p_brks)

brk_locs = get_brk_locations(seq, num_brks, cats, dp)

# just print breakpoint locations
print("breakpoint locations: %s" % (brk_locs))
