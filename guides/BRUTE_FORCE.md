## Brute-forcing a more complex model

### Background

Let's extend our model. In the previous example, we assumed that only two coins were used to create our sequence, with $\theta_1$ and $\theta_2$ probability of heads respectively. This time, our sequence will be created by flipping an unknown number of coins $n_c$ times, each with a different $\theta_c$ probability of heads.

Our definition of a breakpoint stays the same; a location in the sequence where the value of $\theta$ changes. However, this time we have an unknown number of breakpoints $\Kappa$. These breakpoints sub-divide our sequence into $\Kappa + 1$ segments, which each segment having a different $\theta_k$.

> NOTE: $\theta_c$ and $\theta_k$ are equivalent. We just choose to write it as the latter because it makes more sense in our model - we're modelling breakpoints, not coins. This is also the case for $n_k$ which will come up later.

Consider the following sequence where * represents heads and _ represents tails:

```
k=0   | k=1   | k=2
***** | _____ | *****
```

> NOTE: Breakpoints are indicated with |.

- Breakpoints are located at positions 5 and 10 - the value of $\theta$ changes at these points.
- $n_0 = 5$ - the number of elements in the first segment.
- $\theta_0 = 1$ - the probability of heads in the first segment.
- $\theta$ for the sequence between breakpoints $k = 1$ and $k = 2$ is labelled $\theta_1$ - we use the value of $k$ from the preceeding breakpoint.

> NOTE: Since we have $\Kappa + 1$ segments but only $\Kappa$ breakpoints, anything relating to the sequence before the first breakpoint is considered $k = 0$. For example, the value of $\theta$ for the sequence before the first breakpoint is termed $\theta_0$. We can imagine a breakpoint zero that is always present at the start of our sequence.

### Priors

#### Breakpoints

Since $\Kappa$ will be treated as an unknown parameter in our model, we need a prior for it. This time we do have prior assumptions about what values $\Kappa$ might take: we assume that higher numbers of breakpoints are increasingly unlikely. This is expressed as follows:

$
P(\Kappa) = \frac{0.5}{k + 1}
$

#### Segmentations

Another thing that we need to consider in our model is the segmentation of our sequence - the 'segmentation' describes all the possible ways to sub-divide a sequence given $\Kappa$ breakpoints. We have actually come across something similar in our single breakpoint model - $g(a)$. For a sequence of $n$ elements, there are $n + 1$ valid breakpoint locations, and each of those is given equal weight.

$
g(a) = \frac{1}{n + 1}
$

> NOTE: This definition of $g()$ considers the start of the sequence a valid breakpoint location.

This definition of $g(a)$ was used along with the following code in the single breakpoint example:

```py
# g(a) - breakpoint equally likely at each position in the sequence
pri_brk = 1 / (len(seq) + 1)

# loop through all the segmentations (possible breakpoint locations) in our sequence
# NOTE: here we treat the first position as a valid breakpoint location - we will address this later
for pos in range(len(seq)):
  # p = ...

  p_brk = p * pri_brk
```

However, we need to extend this definition to be compatible with an any number of breakpoints. We can use the binomial coefficient to express this: $_nC_k$, where $n$ is the length of our sequence and $k$ is the number of breakpoints:

$
g(a) = \binom{n}{1}^{-1}
$

The two formulae above are equivalent. However, we can use the latter to generalise over any number of breakpoints. Additionally, we want to disallow the beginning of our sequence as a valid breakpoint location. We end up with the following:

$
P(A | \Kappa) = \binom{n - 1}{k}^{-1}
$

Here we define the 'segmentation' of our sequence as $P(A)$ instead of $g(a)$. We also express that the segmentation is dependent on the number of breakpoints in our sequence, thus it becomes $P(A | \Kappa)$.

> NOTE: Much like $g(a)$, $P(A | \Kappa)$ is also a prior in our model.

#### Breakpoints in segmentations

Another thing worth mentioning is the meaning of $A_k$. Given some concrete segmentation $A = a$ of our sequence, $A_k$ is the location of breakpoint $k$ in the sequence.

From our earlier example:

```
k=0   | k=1   | k=2
***** | _____ | *****
```

We can see that $A_0 = 0$, $A_1 = 5$ **for the segmentation displayed above**. If we changed the locations of the breakpoints in the above example, we would end up with a different segmentation and thus, $A_0$ and $A_1$ would have different values.

### Formulae

We can follow 'The basic segmentation model' section of *Bayesian inference on biopolymer models* to obtain the following formulae. You might notice that these are equivalent to $P(A = a, y_{obs})$ and $P(y_{obs})$ in our previous example. The only thing that we've changed is the naming of our variables and the complexity of our model - the process stays the same!

$
P(R|\Kappa = k) = \sum_A \prod_k \frac{\Gamma(\sum_c \alpha_c)}{\prod_c \Gamma(\alpha_c)} \frac{\prod_c \Gamma(n_{k,c} + \alpha_c)}{\Gamma(n_k + \sum_c \alpha_c)}
$

$
P(R) = \sum_{k = 0}^{k_{max}} P(\Kappa = k) P(R | \Kappa = k)
$

> NOTE: $A$ is dependent on the value of $k$ that we want to sample, there is just no nice way to express this in the formula.

#### Notes

- $n_k$ - the number of elements in segment $k$.
- $n_{k,c}$ - the number of elements in segment $k$ that belong to category $c$.

### Calculations

First, we need to define the priors discussed above. This is surprisingly easy. One thing to note is how the dependency of $A$ on $\Kappa$ in $P(A | \Kappa)$ is expressed in code: it translates to a function that takes $\Kappa = k$ as a parameter. We could have done a similar thing before with $g(a)$, taking the breakpoing location $a$ as a parameter.

```py
# the prior for P(K) - priors can be functions too!
def get_pri_num_brks(num_brks):
	p = 0.5 / (num_brks + 1)

	return p
```

```py
from math import comb

# the prior for P(A | K)
# NOTE: we leave out the (n - 1) part of our formula, we adjust for this in
# another part of the code, but it can just as well be included here!
def get_pri_segmentation(num_brks, seq_len):
	p = comb(seq_len, num_brks) ** -1

	return p
```

If we look at our formula above, there are only two differences from earlier examples:

We now sum over all $k$ segments: $\prod_k$. Before, we assumed only one breakpoint (and therefore two segments): $\prod_{i=1}^2$.

Since we want to infer the probability of $k$ breakpoints in our sequence instead of the breakpoint location, we sum over all the possible segmentations: $\sum_A$, where $A$ is dependent on the value of $k$ we choose to sample.

For a single breakpoint, the code is very similar to that in the single breakpoint example:

```py
p = 0

# enumerate all the possible segmentations of our sequence given k = 1
# NOTE: the range(1, ...) here means that we dont allow a breakpoint before the first element
for pos in range(1, len(seq)):
  left = seq[:pos]
  right = seq[pos:]

  # instead of sampling at each position, we now add up all the probabilities
  # and later weigh them by the probability of observing each segmentation
  p += infer_prob_seq(left, cats) * infer_prob_seq(right, cats)

# equivalent of p * g(a) in the previous example
return p * pri_segmentation
```

A similar pattern exists for higher numbers of breakpoints:

```py
# an equivalent example with k = 2 breakpoints
p = 0

# this time we have more segmentations to enumerate
for first_brk in range(1, len(seq)):
  for second_brk in range(first_brk + 1, len(seq)):
    left = seq[:first_brk]
    mid = seq[first_brk:second_brk]
    right = seq[second_brk:]

    # this is different since we now multiply over all k segments instead of just 2
    p += infer_prob_seq(left, cats) * infer_prob_seq(mid, cats) * infer_prob_seq(right, cats)

return p * pri_segmentation
```

I called this the 'brute force' approach for a reason. Firstly, this isn't a fully working example: we would need to create an implementation for every value that $k$ could take. One option is to use recursion to accomplish this without the need for code duplication and infinitely nested for loops. However, you might notice that we soon run into an issue: the segmentations of a sequence increase exponentially with sequence length and breakpoint count. Simply, this approach will not work with large data.

A fully working example can be found in `brute_force.py`.
