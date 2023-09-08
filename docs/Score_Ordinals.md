# Packed Indices for Mastermind Scores

Mastermind guesses are evaluated against the secret and given a score made up of a number of black $b$ and white $w$
hits. Black hits are awarded for every correct color in the correct position, and white hits are awarded for every
correct color in the wrong position. Each colored pin in the secret can be used only once. Scores are typically
represented by the two numbers $b$ and $w$, either as separate values or packed together into a single value. Games have
a number of pins $p$ and colors $c$.

A reasonable human-readable packing for up to 9 pins is $10b + w$, so a score of $b = 2$, $w = 1$ would be 21.

Packing the result into a single byte is reasonable for games with up to 15 pins with $(b \ll 4) | w$, so the same score
would be `0x21`.

These are reasonable representations, but Mastermind scores are sparse. Here are the possible packed scores for a 4 pin
game:

$$[00, 01, 02, 03, 04]\ [10, 11, 12, 13]\ [20, 21, 22]\  [30]\ [40]$$

It is common for algorithms to want to record, say, the total number of guesses which produce each score. See, for
example, Knuth's algorithm in [1]. One could use a 2D array and index with `[b][w]`, resulting in 25 cells for a 4 pin
game, half of them wasted. One could instead use a 1D array and index with the packed decimal scores shown above,
resulting in 41 cells and more waste, or use the packed hex score with 65 cells and even more waste. Even representing
the scores as base- $p$ numbers results in 30% waste.

It's worth noting that the total number of distinct score values is given by[2]:

$$t = {p(p + 3) \over 2}$$

Given the numbers involved are quite small, it's reasonable for most implementations to choose any of the methods above,
even for larger games. In fact, the CPU implementations here do precisely this, trading the simplicity of a sparse array
against the cost of anything more complex.

However, certain kinds of GPU memory are quite limited, and parallelizing interesting gameplay algorithms on the GPU
requires careful use of shared memory. Wasting half of it or more is unacceptable since it's on the order of 48-100KiB
and every thread in a block needs a portion.

Some implementations may choose to use a sparse lookup table to translate scores to a dense range, and others employ
(possibly large) switch statements. Neither of these are particularly good on the GPU either due to warp divergence.

## Dense Packing

Grouping scores by $b$, the number of scores with $b = 0$ is $p + 1$, with $b = 1$ is $p$, and with
$b = 2$ is $p - 1$, and so on until $b = p$ with just a single score.

At this point you may note that the score $b = p - 1$, $w = 1$, is not possible. That would say that all colors in the
guess are correct but only one of them is in the wrong place. That's clearly physically impossible, so there is a hole
in the sequence suggested above. We will temporarily ignore that and assume that such a score is possible.

Given that, the size of each successive group of scores $GS$ with a given black value $b$ is given by:

$$GS_b = p + 1 - b$$

To give all scores an ordinal $O_{b,w}$ we need to know the starting number $GB_b$ of each group, then simply add $w$.

$O_{b,w}=GB_b+w$

Observe that each subsequent group size $GS_i$ decreases by one until $GS_p = 1$, and that $GB_i$ is therefore the sum
of all $GS_0$ to $GS_{i - 1}$.

$$GB_b=\sum_{n=0}^{b-1}GS_n$$

$$GB_b=\sum_{n=0}^{b-1}(p+1-n)$$

$$GB_b=\sum_{n=0}^{b-1}(p+1) - \sum_{n=0}^{b-1}n$$

$$GB_b=b(p+1) - \frac{(b-1)b}{2}$$

In a scoring function which computes the black hits $b$ and all hits $a$, both black and white, the number of white hits
is given by $w = a - b$. Thus,

$$O_{b,w} = b(p + 1) - \frac{(b - 1)b}{2} + w$$

$$O_{b,w} = bp + b - \frac{(b - 1)b}{2} + a - b$$

$$O_{b,w} = \frac{2bp}{2} - \frac{(b - 1)b}{2} + a$$

$$O_{b,w} = \frac{2bp + (1 - b)b}{2} + a$$

$$O_{b,w} = \frac{2bp + b - b^2}{2} + a$$

$$O_{b,w} = \frac{b(2p + 1 - b)}{2} + a$$

It turns out that [isn't too bad to compute at all](https://godbolt.org/z/j7rse3axa), and it saves significant shared
memory on the GPU, enabling much, much better occupancy and thus throughput. This has been factored to favor more work
at compile time, and results in 4 ops: sub from a constant, multiply, divide, add.

Note, however, that the packing is not perfect. The equations above leave room for the impossible score
$b = p - 1$, $w = 1$. This can be adjusted for programmatically, by subtracting $1$ if $b = p$, but it's not worth it in
this case. The extra word of memory is expendable vs. having the GPU execute an `if` statement.

A visual representation is helpful to understanding the above equations. For a 5 pin game:

```
Score:            00 01 02 03 04 05    10 11 12 13 14    20 21 22 23    30 31 32     40 xx        50
Ordinal:           0  1  2  3  4  5     6  7  8  9 10    11 12 13 14    15 16 17     18 19        20
Size vs. first:    0                   -1                -2             -3           -4           -5
Starting ordinal:                      1(p+1) - 0        2(p+1) - 1     3(p+1) - 3   4(p+1) - 6   5(p+1) - 10
```

I.e., subtract the running sum of *'Size vs. first'* from $b(p+1)$.

## References

[1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.

[2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March 2013, arXiv:
1305.1010 [cs.GT]. https://arxiv.org/abs/1305.1010

