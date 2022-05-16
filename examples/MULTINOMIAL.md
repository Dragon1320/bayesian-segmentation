## Extending the definition of a sequence

### Background

So far we have defined a sequence as a series of coin flips. However, this is fairly limiting. To make this algorithm universally applicable, let's define a sequence as composed of $n$ elements where each element can belong to any number of categories $C$.

If we think about it, our coin flip sequence is a simplification of the above definition where the number of categories is limited to 2; $C_1 = heads$ and $C_2 = tails$. We've modelled this so far using the binomial distribution.

However, the binomial distribution is a special case of the multinimial distribution - one where we only have two categories. So we can just use the multinimial distribution instead to extend our model!

So this (binomial with beta prior):

$
P(y_{obs}) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \frac{\Gamma(h + \alpha) \Gamma(t + \beta)}{\Gamma(n + \alpha + \beta)}
$

Becomes this (multinimial with dirichlet prior):

$
P(y_{obs}) = \frac{\Gamma(\sum_c \alpha_c)}{\prod_c \Gamma(\alpha_c)} \frac{\prod_c \Gamma(n_c + \alpha_c)}{\Gamma(n + \sum_c \alpha_c)}
$

#### Notes

- $n_c$ is the number of elements in the sequence that belong to the category $c$
- similarly, $\alpha_c$ is the prior for category $c$
- $\sum_c$ and $\prod_c$ means sum (or multiply) over all possible categories

### Calculations

We conveniently split the calculation into two parts last time. The only thing we need to do is replace the $...$ in the following formula:

$
P(A = a, y_{obs}) = g(a) \prod_{i=1}^2 \left[...\right]
$
