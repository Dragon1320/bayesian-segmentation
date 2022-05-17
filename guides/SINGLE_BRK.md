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

To get an intuitive understanding of the process, let's consider the following sequence of heads (1s) and tails (0s). The breakpoint is indicated with a |. On the left of the breakpoint $P(heads) = \theta_1$, while on the right of the breakpoint $P(heads) = \theta_2$.

```
# case 1
1 1 1 1 1|0 0 0 0 0
left ^ right

# case 2
1 1 1 1 1|0 0 0 0 0
    left ^ right
```

Let's say we split the sequence in case 1 based on the position of ^ into a left and right part. $P(heads)$ in the left sequence will be dependent only on $\theta_1$. However, $P(heads)$ in the right sequence will be dependent on a mixture of $\theta_1$ and $\theta_2$. Therefore, $P(right)$ for any single value of $\theta$ will be low - the sequence cannot be described using a single $\theta$ and is a mixture.

In case 2, $P(heads)$ in both left and right is dependent only on $\theta_1$ and $\theta_2$ respectively. Nether left nor right was created based on multiple values of $\theta$. Therefore, the value of $P(left) P(right)$ will be maximised at the location of a breakpoint - the point at which sequence on either side was created using a single $\theta$.

A fully working example can be found in `single_brk.py`.
