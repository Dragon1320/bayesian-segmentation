from math import comb

import numpy as np
from mpmath import mp, gamma

from prior import p_num_brks

# for generating example data
n1 = 40
n2 = 40
p1 = 0.9
p2 = 0.1

# priors (analogous to a = b = 1)
cats = {
	"_": 1,
	"*": 1,
}

def generate_example_seq(n1, n2, p1, p2):
  seq = np.concatenate([np.random.binomial(1, p1, n1), np.random.binomial(1, p2, n2)])
  # TODO: this is bad code!
  seq = ["_" if c == 0 else "*" for c in seq]

  return seq

def get_pri_num_brks(num_brks):
	p = 0.5 / (num_brks + 1)

	return p

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
	# TODO: optimisation - realistically, this check only needs to be done once
	assert(n == n_counts)

	p_mul = 1

	for cat, pri in cats.items():
		count = counts[cat]

		p_mul *= gamma(count + pri)

	p = (gamma(pri_sum) / pri_gamma_mul) * (p_mul / gamma(n + pri_sum))

	return p










# TODO: new!
# TODO: consistent len
def init_dp_array(seq, kmax, cats):
	# k starts at 0, account for this in the array size
	# we need the big boi floats for the calcs were doing owo
	dp = np.ndarray(shape=(kmax + 1, len(seq) + 1, len(seq) + 1), dtype=mp.mpf)

	# k = 0
	for j in range(0, len(seq)):
		for v in range(j, len(seq)):
			sub_seq = seq[j:v + 1]

			p =  infer_prob_seq(sub_seq, 0, len(sub_seq), cats)

			# these values should be treated the same as seq[j:v]
			# namely, the element at idx v is not included
			dp[0][j][v + 1] = p

	return dp

# TODO: new!
# TODO: consistent naming
def infer_prob_num_brks(seq, kmax, cats, dp):
	# running this for anything else is dum
	assert(kmax >= 1)
	# this is important as we can only have as many breakpoints as n + 1 elements
	# ie. we cannot have 2 breakpoints in a sequence composed of 2 elements
	assert(kmax < len(seq))

	# TODO: assert that dp was initialised with the same sequence

	# now we use the stuff to calculate the other stuff
	# very descriptive ik, im somewhat of a poet

	# k = 1 -> k = kmax
	for k in range(1, kmax + 1):
		for j in range(0, len(seq) - k):
			p = 0

			# TODO: link some diagram that explains how im calculating these
			for s in range(j + 1, len(seq) - (k - 1)):
				left = dp[0][j][s]
				right = dp[k - 1][s][len(seq)]

				p += left * right

			dp[k][j][len(seq)] = p






# TODO: new!
def p_num_brks_given_sequence(seq, kmax, cats, dp):
	# i could incorporate this stuff into p_sequence_given_num_brks but its cheap (only runs kmax times)
	# and this way its more clear what im doing above - which is already difficult to understand
	p_brks = []

	# TODO: ret k = 0 as well?
	# TODO: is the final p calc correct? k set to the correct val everywhere?

	# in case someone wanted to re-use a sub-sequence with the same dp array,
	# len(seq) is used instead of -1
	for k in range(0, kmax + 1):
		# we set the length as len(seq) - 1, as otherwise it would allow for breakpoints at the start
		# of the sequence (ie. before the first element), which is not what we want
		p = dp[k][0][len(seq)] * p_segmentation_given_num_brks(len(seq) - 1, k)

		# prior
		pk = p_num_brks(k)

		p_brks.append(p * pk)

	# marginal probability
	pr = sum(p_brks)
	p_brks = [p / pr for p in p_brks]

	return pr, p_brks

