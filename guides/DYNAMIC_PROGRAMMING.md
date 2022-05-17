## Optimisations with dynamic programming

### Background

This describes how we can complete the calculations from our brute-force example in a more optimised way. Without this, we would not be able to apply this algorithm to real-world data.

Note that this approach diverges significantly from *Bayesian inference on biopolymer models*. The algorithm described in the paper doesn't work. The only similarity between my approach and theirs is that both methods use some kind of 'dynamic programming' algorithm.

In brief, dynamic programming is a programming approach where one needs to only perform expensive calculations for some base-case. Any further calculations can be composed from the values obtained from our base-case calculations. These compositions are significantly cheaper to perform than performing non-base-case calculations from scratch. 

### Algorithm

Consider a sequence of 5 elements that contains $k = 2$ breakpoints. Elements are indicated by _ and breakpoints are indicated by |. We can group segmentations based on the location of the first breakpoint. In this case, this results in 3 groups.

```
# group 1
_|_|_ _ _
_|_ _|_ _
_|_ _ _|_

# group 2
_ _|_|_ _
_ _|_ _|_

# group 3
_ _ _|_|_
```

Referring to the brute-force example, we can calculate the probaility of group 1 as such:

```py
p = 0

for segmentation in group_1:
  # note that the location of the first breakpoint in group 1 is constant
  # therefore, seg_1 will be the same for all iterations of this loop
  p += seg_1 * seg_2 * seg_3

return p
```

> NOTE: We're ignoring priors for now.

Since we know that for any given group, the left-most segment will be constant, we can simplify:

```py
p = 0

for segmentation in group_1:
  p += seg_2 * seg_3

# this is ok, multiplication is commutative
return p * seg_1
```

With this simplification, we've split the calculation into two steps:
- Calculate the probability of sequence left of the first breakpoint for $k = 0$.
- Calculate the probability of sequence right of the first breakpoint for $k = k - 1$.

To demonstrate the second step, let's take the sequence right of the first breakpoint in group 1: `_ _ _ _`. Since there are $k = 2$ breakpoints in the above example, let's look at all the possible segmentations of this sequence for $k = k - 1$:

```
# example set
_|_ _ _
_ _|_ _
_ _ _|_
```

And to calculate the probability of those segmentations:

```py
p = 0

for segmentation in example_set:
  # the variable names seg_2 and seg_3 were kept on purpose to draw parallels to previous examples
  p += seg_2 * seg_3

# this is exactly the same as the simplified calculation for group 1 above
# the only difference is that we dont multiple by seg_1
return p
```

This example is very important as it shows that the calculation for $k = 2$ can be composed from a $k = 0$ and $k = 1$ part. Similarly when finding the probability of some sequence with $k = 1$ breakpoints, we can split this calculation into two $k = 0$ parts. Since calculations involving higher values of $k$ can be split into a $k = 0$ and a $k - 1$ component, we only need to perform calculations for $k = 0$ and use these to compose calculations for higher values of $k$.

### Calculations

This approach solves all of our issues from the brute force example. Firstly, since there is a pattern to our calculations, we don't need to duplicate code and nest for loops.

```py
for first_brk in range(1, len(seq)):
  for second_brk in range(first_brk + 1, len(seq)):
    # loop hell insues...
```

Instead, we can start from $k = 0$ and build our way up to higher numbers of breakpoints using the following formula and code:

$
P(sequence | k) = P(left | 0) P(right | k - 1)
$

```py
# calculate all values of k, from k = 1 to some maximum
# this assumes that all values for k = 0 have already been pre-calculated
for k in range(1, kmax + 1):

  # here j takes on the value of each position in our sequence, but can also represents our groups from earlier
  # incrementing j here represents moving the position of the first breakpoint progressively through the sequence
  # we stop k away from elements from the end since we cant fit k breakpoints in a sequence that is <= k elements long
  for j in range(0, len(seq) - k):
	  p = 0

	  for s in range(j + 1, len(seq) - (k - 1)):
		  left = dp[0][j][s]
		  right = dp[k - 1][s][len(seq)]

		  p += left * right

    # we store our pre-calculated in some 3 dimensional array - this will be explained later
    dp[k][j][len(seq)] = p
```

This also conveniently solves our performance issues; we simply compose calculations from our base case of $k = 0$. 
- We don't need to loop through all the possible segmentations for higher $k$ values.
- All the calculations in this loop are cheap to perform. This is not the case when calculating these values using our formulae from earlier examples.

Of course, the above code assumes that all calculations for $k = 0$ have been completed - we still need to take care of those. In fact, the dynamic programming approach above is so efficient that performing all the calculations for $k = 0$ is now the slowest step in our algorithm!

One thing to note: since we split our sequence into a left and right component in the above calculation, we need to pre-calculate all possible left and right sub-sequences for the $k = 0$ case. Likewise, you might have noticed that in the dynamic programming algorithm above, we also calculate sub-sequences for all other values of $k$.

```py
from mpmath import mp
import numpy as np

# this 3d array will hold our pre-calculated values, and is indexed as follows:
# k = breakpoint count
# j = start of subsequence
# v = end of subsequence

# k starts at 0, account for this in the array size
# we need the big boi floats for the calcs were doing owo
dp = np.ndarray(shape=(kmax + 1, len(seq) + 1, len(seq) + 1), dtype=mp.mpf)

# calculate the k = 0 base case for all sub-sequences
for j in range(0, len(seq)):
	for v in range(j, len(seq)):
		sub_seq = seq[j:v + 1]

		p =  infer_prob_seq(sub_seq, 0, len(sub_seq), cats)

		# the indexing in the dp array works the same as python lists
		# this means that dp[k][start][end] includes the value at start, but does not include the value at end
		dp[0][j][v + 1] = p
```

Our dynamic programming algorithm requires one last component to tie everything together. You might have noticed that we have ignored our priors up until now - particularly $P(A | \Kappa)$ which has dissappeared from similar-looking brute-force examples. Since we compose our calculations from previously calculated values, applying a prior at the beginning will result in that prior being applied to any downstream calculations - not what we want.

```py
# an array holding 
p_brks = []

# in case someone wanted to re-use a sub-sequence with the same dp array,
# len(seq) is used instead of -1
for k in range(0, kmax + 1):
  # dp[k][0][len(seq)] holds the probability of the full sequence given k breakpoints
  # we apply the prior here - P(A | K)
	p = dp[k][0][len(seq)] * get_pri_segmentation(len(seq) - 1, k)

	# we also our other prior - P(K)
	pk = p_num_brks(k)

	p_brks.append(p * pk)

# marginal probability
pr = sum(p_brks)
p_brks = [p / pr for p in p_brks]
```

A fully working example can be found in `dynamic_programming.py`.

> NOTE: If we set $kmax = 2$ (the max value of $k$ that we handle in the brute-force approach), the results will be exactly the same!
