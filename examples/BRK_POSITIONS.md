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
