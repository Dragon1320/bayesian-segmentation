# bayesian-segmentation
A bayesian sequence segmentation algorithm.

### Sequence

$
P(\theta_1 | y_{obs}) = \frac{\Gamma(n + \alpha + \beta)}{\Gamma(h_n + \alpha) \Gamma(n - h_n + \beta)}\theta_1^{h_n + \alpha - 1}(1 - \theta_1)^{t_n + \beta - 1}
$

$n$ - length of sequence

$h_n$ - number of heads

$t_n$ - number of tails

### Single breakpoint

$
P(A = a, y_{obs}) = g(a) \prod_{i=1}^2 \left[\frac{\Gamma(\alpha_i + \beta_i)}{\Gamma(\alpha_i) \Gamma(\beta_i)} \frac{\Gamma(h_i + \alpha_i) \Gamma(t_i + \beta_i)}{\Gamma(h_i + t_i + \alpha_i + \beta_i)}\right]
$

$
P(y_{obs}) = \sum_a P(A = a, y_{obs})
$

### Multinomial

$
P(y_{obs}) = \frac{\Gamma(\sum_d \alpha_d)}{\prod_d \Gamma(\alpha_d)} \frac{\prod_d \Gamma(n_d + \alpha_d)}{\Gamma(n + \sum_d \alpha_d)}
$

### Brute force

$
P(\Kappa) = \frac{0.5}{\kappa + 1}
$

$
P(A | \Kappa) = \binom{N - 1}{\kappa}^{-1}
$

> NOTE: this means we cant have breakpoints on either end (this makes more sense imo)

$
P(R_{[j:len)} | k) = \sum_{s = j + 1}^{len - k - 1} P(R_{[j:s)} | 0) P(R_{[s:len)} | k - 1)
$

> NOTE: j should range [0:len - k)

> NOTE: this isnt quite correct...
