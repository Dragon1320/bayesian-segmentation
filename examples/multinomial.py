from mpmath import mp, gamma

import numpy as np
import matplotlib.pyplot as plt

# for generating example data
n1 = 20
n2 = 20
p1 = 0.9
p2 = 0.1

# priors - we now specify a prior per category
# this time instead of 1s and 0s, we have a sequence of _s and *s

# にゃ～
cats = {
	"_": 1,
	"*": 1,
}

# generate an example sequence of _s and *s instead
def generate_example_seq(n1, n2, p1, p2):
  seq = np.concatenate([np.random.binomial(1, p1, n1), np.random.binomial(1, p2, n2)])
  seq = ["_" if c == 0 else "*" for c in seq]

  return seq

# probability of observing the given sequence - this is the only part that has changed (in line with the formulae shown)
# again, were only calculating values based on an observed sequence and plugging them into a formula
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

# this part remains exactly the same
def infer_prob_brk(seq, pos, pri_brk, cats):
  n = len(seq)

  left = seq[:pos]
  right = seq[pos:]

  p = 1

  for sub_seq in [left, right]:
	  p *= infer_prob_seq(sub_seq, cats)

  return p * pri_brk

# calculations
seq = generate_example_seq(n1, n2, p1, p2)
print("sequence: %s" % ("".join(seq)))

# g(a) - breakpoint equally likely at each position in the sequence
pri_brk = 1 / (len(seq) + 1)

p_pos = []

for pos in range(len(seq)):
  p_brk = infer_prob_brk(seq, pos, pri_brk, cats)

  p_pos.append(p_brk)

# calculate P(y_obs)
p_obs = sum(p_pos)

# plotting
scale_x = np.arange(0, len(seq))
scale_y = []

for x in scale_x:
  y = p_pos[x] / p_obs

  scale_y.append(y)

plt.title("finding a single breakpoint")
plt.xlabel("position in sequence")
plt.ylabel("probability of breakpoint")

plt.bar(scale_x, scale_y)
plt.savefig("multinomial.png")
