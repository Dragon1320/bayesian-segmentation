## Brute-forcing a more complex model

### Background

Let's extend our model. Our sequence will now be composed from _s and *s instead of 1s and 0s. Much like before, the frequency of _s and *s in any given section of the sequence is governed by some $\theta$.

The above sequence has an unknown number of breakpoints $\Kappa$. Each segment in the sequence is separated by a breakpoint, and each segment has a different value of $\theta$.

To demonstrate the above, consider the following sequence:

`*****|_____`

The length of the sequence is $n = 10$.

There is a single breakpoint in this sequence; $\Kappa = 1$. The breakpoint is indicated with `|`.

The 'segmentation' of this sequence describes the possible locations of breakpoints. One of the possible segmentations of the above sequence is shown (breakpoint indicated with `|`). However, the above sequence has $C_{9,1}$ possible breakpoint locations.

- The sequence is composed of $k + 1$ parts, with each part having a different $\theta_k$
- We will call the 'segmentation' of the sequence $A$, where $A_k$ is the location of breakpoint $k$ in a given segmentation.

In the segmentation shown above, $A_k = 5$.

The following formula can be found in 'The basic segmentation model' section of *Bayesian inference on biopolymer models*:

Priors:

$
P(\Kappa) = \frac{0.5}{\kappa + 1}
$

$
P(A | \Kappa) = \binom{n - 1}{\kappa}^{-1}
$

Calculations:

$
P(R|\kappa = k) = \sum \prod \frac{\Gamma(\sum_c \alpha_c)}{\prod_c \Gamma(\alpha_c)} \frac{\prod_c \Gamma(n_{k,c} + \alpha_c)}{\Gamma(n_k + \sum_c \alpha_c)}
$

$
P(R) = \sum_{k = 0}^{k_{max}} P(\kappa = k) P(R | \kappa = k)
$

#### Notes

- The prior for $P(A | \Kappa)$ is different - we disallow a breakpoint before the first element (because that makes more sense).
- $n_k$ - the number of elements in segment $k$
- $n_{k,c}$ - the number of elements in segment $k$ that belong to category $c$

### Calculations

We can draw a surprising number of parallels to the single breakpoint example.

Firstly, $g(a)$ in the single breakpoint example is equivalent to $P(A | \Kappa)$. If we recall:

$
g(a) = \frac{1}{n + 1}
$

This means that in a given sequence of length $n$, every breakpoint location is equally likely. We define the segmentation of a sequence with $k = 1$ breakpoints as: every possible location of that breakpoint. This is equivalent to: $C_{n,1}$. We can abstract this to any number of breakpoints with $C_{n,k}$ which brings us to our current definition of $P(A | \Kappa)$

```py
from math import comb

def get_pri_segmentation(num_brks, seq_len):
	p = comb(seq_len, num_brks) ** -1

	return p
```

Secondly, since we cannot assume the number of breakpoints to be $k = 1$, we need to treat $k$ as a parameter in our model. Therefore we need a prior for it:

$
P(\Kappa) = \frac{0.5}{\kappa + 1}
$

This prior expresses a belief that higher numbers of breakpoints are increasingly unlikely.

```py
def get_pri_num_brks(num_brks):
	p = 0.5 / (num_brks + 1)

	return p
```

Thirdly, in the single breakpoint example, we assumed the number of breakpoints to be $k = 1$, which meant the following code was sufficient:

```py
p = 0

# NOTE: the range(1, ...) here means that we dont allow a breakpoint before the first element
for pos in range(1, len(seq)):
  left = seq[:pos]
  right = seq[pos:]

  p += infer_prob_seq(left, cats) * infer_prob_seq(right, cats)

# equivalent of p * g(a) in the previous example
return p * pri_segmentation
```

However, we don't know the number of breakpoints in this example. We need to provide calculations for other numbers of breakpoints (eg. $k = 0$ or $k = 2$).

```py
p = 0

# an equivalent example with k = 2 breakpoints
for first_brk in range(1, len(seq)):
  for second_brk in range(first_brk + 1, len(seq)):
    left = seq[:first_brk]
    mid = seq[first_brk:second_brk]
    right = seq[second_brk:]

    p += infer_prob_seq(left, cats) * infer_prob_seq(mid, cats) * infer_prob_seq(right, cats)

return p * pri_segmentation
```
