# Mastermind on the GPU

The core of most interesting Mastermind algorithms involves scoring all possible codewords, call the this All Set (AS),
against a diminishing set of possible solutions which we'll call the Possible Set (PS). 
This is *O(n<sup> 2</sup>)* in the total number of codewords, which is *n = c<sup> p</sup>*. This has to be done once
for each turn in a game, and thus many, many times when playing all games.

My implementation pushes this core algorithm over to the GPU at each stage of playing a game, and falls back to a
regular CPU implementation for very small AS & PS combinations.

## AS is Constant

AS is not really constant, as it seems worthless to consider a codeword which has already been played. But for any
interesting game size the maximum number of turns is a tiny, tiny fraction of AS and can be ignored. No previously 
played codeword will rise to the top in any valuation, so the only cost to keeping AS constant is really the extra time
needed to consider these few worthless codewords.

There is, however, extreme value to keeping AS constant in the GPU implementation. We need the codewords and pre-computed
color counts for AS in GPU memory, and keeping AS constant allows us to load our largest dataset just once and re-use 
it for every game played. This is a big savings. 

## Computing Subset Sizes

The first step is to score every element of AS against every element of PS. This is done by creating a GPU thread for
every element of AS and letting it loop over PS, computing scores. We don't actually need to remember the scores, all
we need is to find how many of each score we saw. This effectively subsets PS by score value, one possible subset for
every score value. This means that we need a counter for each possible score value large enough to count to the total
number of codewords. A 32 bit counter is used, so for the classic game of 4p6c we need 15 counters.

That doesn't sound like a lot of counters, and it isn't. However, it does grow as game sizes grow, and keeping the 
counters as thread local storage (i.e., consuming scalar or vector registers) quickly leads to a GPU kernel with 
low occupancy due to register pressure. 

Instead, the subset sizes are kept in threadgroup memory. While this memory is indeed shared among all threads in a 
theradgroup, this implementation doesn't do any sharing. It's just a fast buffer of which each thread gets a unique
slice as temporary storage.

## Compact Scores

Threadgroup memory is quite limited in Metal on macOS: just 32kb. In order to do a reasonable amount of work per threadgroup,
we use the compact score values described in [Packed Indices for Mastermind Scores](docs/Score_Ordinals.md). These 
score values cost a little more to compute, but it's very minor. And these strange score values never escape the GPU,
only being used to indices into the subset sizes, so it's okay that they're completely different from scores used
elsewhere in the program. This allows us to easily have enough threadgroup memory for large games with 64 threads per group.
That's a nice threadgroup size for my AMD GPU running under Metal: the SIMD execution width is 32, and every compute unit (CU)
holds two SIMD units.

## Consuming Subset Sizes

Once the sizes are computed, each thread consumes them in order to produce a valuation for each member of AS depending
on the specifics of the algorithm in question. E.g., for Most Parts it counts the number of subsets, and for Kunth's it
finds the largest subset size, etc. The result is placed in a slot in the output buffer, one slot per member of AS.

## Fully Discriminating Guesses Optimization

There is a good optimization due to Ville[2] in which we can notice if a member of AS produces one subset for ever 
member of PS. Such guesses are called "fully discriminating", as they will tell us precisely which member of PS is 
the solution. We have each thread keep track of whether or not a codeword is fully discriminating, but the question is
how to tell the CPU so it can play the guess quickly while avoiding as much work as possible. A bit per member of AS 
doesn't save much time. And we only care about one such guess, not all, and it is traditional to pick the lexically first
guess in the face of multiple options.

In this implementation, each SIMD group with one or more interesting guesses does a SIMD group wide reduction, using
simd_min(), to determine the best guess for the entire group. Then the first thread in the group places it in a per-group
output slot. The CPU can then quickly run over this much smaller number of values and pick the first non-zero element
it finds and play it. This avoids looking at the full results for AS and ends up saving quite a bit of time.

## Scoring Function on the GPU

The scoring function used is based on the hand-vectorized version in [codeword.cpp](../codeword.cpp). The first portion,
computing B with the constant operations on a single 32 bit value was kept unchanged. Happily, GPU's provide popcount.

The second part changed a bit. Metal provides vector data types, like `uint4` and `uchar4`, and automatically turns common
operations on them like addition and minimum into vector ops. Minimums of the pair of 16 color counts are taken with
four vector ops on `uchar4`'s, then the mins are reduced with a combination of a few vector adds followed by a couple
of scalar additions. It would have been nice to have full-width min or add operations over the full possible vector width
instead of being limited to 32 bits via `uchar4`. I don't have enough insight into the final code generation on the GPU
to know if the compiler was smarter about that in the end.

## References

[1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.

[2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March 2013, arXiv:1305.1010 [cs.GT]. https://arxiv.org/abs/1305.1010
