## Sampling an unknown variable

### Background

Let's say we have a coin that we flip $n$ times. Each flip will result in either heads or tails.

- $n$ = length of sequence
- $h$ = number of heads
- $t$ = number of tails

> NOTE: $n = h + t$

```py
# we can represent this assumption/invariant in python!
assert(n == heads + tails)
```

However, when the coin is flipped, we don't know the probability of obtaining heads/tails. Let's treat this probability as an unknown variable $\theta$.

- $P(heads) = \theta$
- $P(tails) = 1 - \theta$

We can infer the value of $\theta$ by following 'A coin example' in *Bayesian inference on biopolymer models*. This formula represents the final calculation:

$
P(\theta | y_{obs}) = \frac{\Gamma(n + \alpha + \beta)}{\Gamma(h + \alpha) \Gamma(n - h + \beta)}\theta^{h + \alpha - 1}(1 - \theta)^{t + \beta - 1}
$

#### Notes

- $y_{obs}$ represents the observed data (length of sequence and number of heads/tails)
- $\alpha$ and $\beta$ are priors, setting $\alpha = \beta = 1$ will give us a uniform distribution (uninformative prior)

### Calculations

Since $P(\theta)$ is a continuous function defined on the interval between 0 and 1, we need to define a step size (i.e. we will sample this function every step size)

```py
i = 0.05

# include both ends (i.e. 0 and 1)
scale_x = np.arange(0, 1 + i, i)

for x in scale_x:
  # calculate the value of theta for for this x...
```

Given some random sequence of 1s (heads) and 0s (tails), we can calculate all the required parameters (n, h, t)

```py
n = len(seq)

heads = (seq == 1).sum()
tails = (seq == 0).sum()
```

Once we have all the information about our observed data, we can use the formula above to calculate the probability of $\theta$ = x.

```py
from mpmath import gamma

# pri_a and pri_b are priors
p = ((gamma(n + pri_a + pri_b)) / (gamma(heads + pri_a) * gamma(n - heads + pri_b))) * (x ** (heads + pri_a - 1)) * ((1 - x) ** (tails + pri_b - 1))
```

In more intuitive terms: we loop through all the possible values of $\theta$ and get the probability of observing each value. This probability should peak at the most likely value for $\theta$.

A fully working example can be found in `sequence.py`.
