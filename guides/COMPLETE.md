# Bayesian sequence segmentation

The following guide goes through all the steps required to implement a bayesian sequence segmentation algorithm based on *Bayesian inference on biopolymer models*.

## Sampling an unknown variable

### Background

Let's say we have a coin that we flip $n$ times. Each flip will result in either heads or tails.

- $n$ - length of sequence
- $h$ - number of heads
- $t$ - number of tails

> NOTE: $n = h + t$

```py
# we can represent this assumption/invariant in python!
assert(n == heads + tails)
```

However, when the coin is flipped, we don't know the probability of obtaining heads/tails. Let's treat this probability as an unknown variable $\theta$.

- $P(heads) = \theta$
- $P(tails) = 1 - \theta$

We can infer the value of $\theta$ given a complete sequence of heads/tails produced by our coin. Following 'A coin example' in *Bayesian inference on biopolymer models*, this formula represents the final calculation:

$
P(\theta | y_{obs}) = \frac{\Gamma(n + \alpha + \beta)}{\Gamma(h + \alpha) \Gamma(n - h + \beta)}\theta^{h + \alpha - 1}(1 - \theta)^{t + \beta - 1}
$

#### Notes

- $y_{obs}$ represents the observed data (length of sequence and number of heads/tails).
- $\alpha$ and $\beta$ are priors, setting $\alpha = \beta = 1$ will give us a uniform distribution (uninformative prior).

> NOTE: Priors represent any knowledge/assumptions we have about the data. An uninformative prior means we don't have any prior knowledge and our inference is fully based on the observed data. Priors can also be thought of as user-supplied parameters that alter the behaviour of our algorithm.

### Calculations

Since the PDF of $P(\theta)$ is continuous and defined on the interval between 0 and 1, we need to define a step size (i.e. we will sample this function every step size).

```py
i = 0.05

# include both ends (i.e. 0 and 1)
scale_x = np.arange(0, 1 + i, i)

for x in scale_x:
  # calculate the value of theta for for this x
  # x will be incremented by 0.05 each iteration
```

Given some random sequence of 1s (heads) and 0s (tails), we can calculate all the required parameters (n, h, t).

```py
n = len(seq)

heads = (seq == 1).sum()
tails = (seq == 0).sum()
```

Once we have all the information about our observed data, we can use the formula above to calculate the probability of $\theta$ = x.

```py
from mpmath import gamma

# pri_a and pri_b are priors
# we use the values for n/heads/tails that we calculated earlier
p = ((gamma(n + pri_a + pri_b)) / (gamma(heads + pri_a) * gamma(n - heads + pri_b))) * (x ** (heads + pri_a - 1)) * ((1 - x) ** (tails + pri_b - 1))
```

In more intuitive terms: we loop through all the possible values of $\theta$ and get the probability of observing each value. This probability should peak at the most likely value for $\theta$.

A fully working example can be found in `sequence.py`.

## Finding a single breakpoint

### Background

Building upon the previous example, let's say we have a sequence created by two different coins; $c_1$ and $c_2$. Each coin is flipped $n_c$ times. The probability of observing heads/tails after flipping each coin in different; $\theta_c$.

- $n = n_1 + n_2$ - length of the sequence
- for the first $n_1$ elements, $P(heads) = \theta_1$
- for the next $n_2$ elements, $P(heads) = \theta_2$

There is a 'breakpoint' after the first $n_1$ elements in the sequence where the probability of heads/tails changes from $\theta_1$ to $\theta_2$. Our goal is to find the position of this breakpoint.

Following 'Two types of coins: Bayesian segmentation' in *Bayesian inference on biopolymer models*, we can complete the required calculations using these formulae:

$
P(A = a, y_{obs}) = g(a) \prod_{i=1}^2 \left[\frac{\Gamma(\alpha_i + \beta_i)}{\Gamma(\alpha_i) \Gamma(\beta_i)} \frac{\Gamma(h_i + \alpha_i) \Gamma(t_i + \beta_i)}{\Gamma(h_i + t_i + \alpha_i + \beta_i)}\right]
$

$
P(y_{obs}) = \sum_a P(A = a, y_{obs})
$

#### Notes

- This time we break the calculation into two parts; First we calculate $P(A = a, y_{obs})$ and $P(y_{obs})$. Then we use Baye's rule to calculate $P(A = a | y_{obs})$. This approach could also have been used for the previous example.

- $g(a)$ is a prior representing the probability of observing a breakpoint at some position $a$ in the sequence.

```py
# g(a) - we set all breakpoint positions to be equally likely
pri_brk = 1 / (seq_len + 1)
```

- Summations $\sum$ and products $\prod$ translate to loops in python.

### Calculations

This time the unknown that we want to infer is the breakpoint location. Similarly to inferring $\theta$ in the previous example, we sample our distribution at every sequence location.

```py
# NOTE: here we can have a breakpoint at position 0 (before the first element)
# this is something we will consider in future iterations of the model
for pos in range(len(seq)):
  # probability of breakpoint at some position in the sequence
  p_brk = infer_prob_brk(seq, pos, pri_brk, pri_a, pri_b)
```

Much like last time, we need to calculate the probability of observing some sequence of heads/tails. However, this time we're not interested in inferring the values of $\theta$. In fact, having multiple unknown parameters in our equation makes computation much more difficult. To solve this, we can integrate over all the possible values of $\theta$ to express it in terms of the remaining unknown variables. This trick is often used in Bayesian statistics to remove 'nuisance parameters'. Such integration results in the equation we see above.

```py
from mpmath import gamma

# this looks different from last time - we've 'integrated out' the theta parameters
p = ((gamma(pri_a + pri_b)) / (gamma(pri_a) * gamma(pri_b))) * ((gamma(heads + pri_a) * gamma(tails + pri_b)) / (gamma(n + pri_a + pri_b)))
```

We've chosen to split the calculation into two parts for the purpose of code clarity;

We have a function that calculates the inner part of the above equation. This prevents code duplication since we need to complete this calculation multiple times - it's part of a loop.

$
\frac{\Gamma(\alpha_i + \beta_i)}{\Gamma(\alpha_i) \Gamma(\beta_i)} \frac{\Gamma(h_i + \alpha_i) \Gamma(t_i + \beta_i)}{\Gamma(h_i + t_i + \alpha_i + \beta_i)}
$

The second function handles the loop and applies our prior.

$
P(A = a, y_{obs}) = g(a) \prod_{i=1}^2 \left[...\right]
$

```py
# outer function...
n = len(seq)

left = seq[:pos]
right = seq[pos:]

# since were doing a loop product, we initialise p = 1,
# if this was a loop summation, we would instead initialise p = 0
p = 1

for sub_seq in [left, right]:
  # call the inner function in a loop
  # notice how this loops twice as defined in our formula
  p *= infer_prob_seq(sub_seq, pri_a, pri_b)

# we apply the prior at the end (in the outer function)
return p * pri_brk
```

### TODO: this needs changed
### TODO: better diagram style

### Explanation

To get an intuitive understanding on the process, let's consider the following scenario. The true breakpoint is marked in blue.

Here we calculate the probability of a breakpoint at the red mark. The probability of heads/tails in the sequence left of the mark will be $\theta_1$. However, in the sequence right of the mark, this will be a mixture of $\theta_1$ and $\theta_2$. Therefore, for any given $\theta$, the probability of observing the sequence on the right will be low.

The maximum value of $P(left) * P(right)$ can be obtained when both $left$ and $right$ where created from a single value of $\theta$ respectively (and not a mixture of $\theta_1$ and $\theta_2$).

This maximum value will be the location of our breakpoint - where the value of $\theta$ changes.

![](https://cdn.discordapp.com/attachments/209040403918356481/975808219932934144/unknown.png)

A fully working example can be found in `single_brk.py`.

## Extending the definition of a sequence

### Background

So far we have defined a sequence as 'a series of coin flips'. However, this is fairly limiting. To make our algorithm universally applicable, let's re-define our sequence as 'a series of $n$ elements, with each element belonging to one of a number of categories $c$'.

If we think about it, our coin flip sequence is a simplification of the above definition where the number of categories is limited to 2; $c_1 = heads$ and $c_2 = tails$. We've modelled this so far using the binomial distribution.

However, the binomial distribution is a special case of the multinimial distribution - one where we only have two categories. So we can just use the multinimial distribution instead to extend our model!

So this (binomial with beta prior):

$
P(y_{obs}) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \frac{\Gamma(h + \alpha) \Gamma(t + \beta)}{\Gamma(n + \alpha + \beta)}
$

Becomes this (multinomial with dirichlet prior):

$
P(y_{obs}) = \frac{\Gamma(\sum_c \alpha_c)}{\prod_c \Gamma(\alpha_c)} \frac{\prod_c \Gamma(n_c + \alpha_c)}{\Gamma(n + \sum_c \alpha_c)}
$

#### Notes

- $n_c$ is the number of elements in the sequence that belong to category $c$.
- Similarly, $\alpha_c$ is the prior for category $c$.
- $\sum_c$ and $\prod_c$ means sum (or multiply) over all possible categories.

### Calculations

We conveniently split the calculation into two parts last time. The only thing we need to do is to replace the $...$ in the following formula:

$
P(A = a, y_{obs}) = g(a) \prod_{i=1}^2 \left[...\right]
$

A fully working example can be found in `multinomial.py`.

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

## Finding breakpoint locations

### Background

Since we can now correctly infer the number of breakpoints in a sequence, let's work on finding their positions. Again, this diverges significantly from *Bayesian inference on biopolymer models*, but this time the general idea behind the method is the same.

### Algorithm

Imagine a sequence of 10 elements with $k = 2$ breakpoints. Elements are represented by _ and breakpoint locations are indicated with |. Knowing the value of $k$, we need to find the locations of those breakpoints in the sequence.

```
_ _ _|_ _ _ _|_ _ _
```

Let's look back to how we found the location of a single breakpoint in an earlier example; the probability $P(sequence) = P(left) P(right)$ should be maximised at the position of our breakpoint. Similarly, in order to find the first breakpoint in the above example, we can find the position where $P(sequence | k) = P(left | 0) P(right | k - 1)$ is maximised.

As an example of why this works, consider the following. In this case, since the sequence has $k = 1$ breakpoints, our calculations looks as follows:

$
P(sequence | 1) = P(left | 0) P(right | 0)
$

```
# case 1
_ _ _ _ _ _ _|_ _ _
left ^ right

# case 2
_ _ _ _ _ _ _|_ _ _
        left ^ right
```

If we perform this calculation with left/right dictated by the position indicated in case 1: left will have 0 breakpoints, but right will not have 0 breakpoints - $P(right | 0)$ will be low. Alternatively, if we use the point indicated in case 2: both left and right will have 0 breakpoints, therefore both $P(left | 0)$ and $P(right | 0)$ will be high.

As described above, this can be generalised to $k$ breakpoints by finding the position at which $P(sequence | k) = P(left | 0) P(right | k - 1)$ is maximised. When we identify the first breakpoint, we can take the subsequence from breakpoint 1 to the end of the sequence and repeat the process with $k - 1$ until we have found all the breakpoints.

### Calculations

Conveniently, we've precalculated any values we might need in the dynamic programming step! The following code samples our sequence for breakpoints as described above using these pre-calculated values.

```py
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
  for j in range(m, len(seq) - q):
    seq_len_right = len(seq) - j
    pri_segmentation_right = get_pri_segmentation(seq_len_right - 1, q - 1)

    # pri_segmentation for left will always be 1 (x choose 0 == 1)
    left = dp[0][m][j]
    right = dp[q - 1][j][len(seq)] * pri_segmentation_right

    # we dont care about P(y_obs) - its just used for normalising our probabilities
    p_seg.append(p)

  # move m to the next breakpoint weve found
  m = m + np.argmax(p_seg)
  brk_locs.append(m)
```

A fully working example can be found in `brk_positions.py`.
