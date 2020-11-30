# Mastermind

This repo contains code to play the game of Mastermind with various numbers of color and pins using various well-known 
algorithms. There are scalar, vectorized, and GPU variants of both the scoring functions and the gameplay.

I was motivated to dig into Mastermind through volunteer work with APCS high school students taught 
by [Michael Miyoshi](https://github.com/MichaelTMiyoshi) at Cedarcrest High School. They are given an assignment to write the scoring
function for Mastermind, which is quite a challenge for new students to get right. I had never heard of the game before,
and I found attempting to play all games efficiently to be a pretty fun challenge myself.

This repo holds most of my work on playing all games of various sizes quickly:

- Multiple versions of the scoring function, including a vectorized version using SSE2. See the comments in [codeword.cpp](codeword.cpp)
for details, and Compiler Explorer links showing the generated code.
- The following algorithms for playing all games. See [strategy.hpp](strategy.hpp) and [subsetting_strategies.hpp](subsetting_strategies.hpp) for more details.
  - First consistent
  - Random consistent
  - Knuth
  - Most Parts
  - Expected Size
  - Entropy
 - A GPU implementation which runs the scoring function and the inner loops of Knuth, Most Parts, etc.
   - This is implemented using Apple's Metal API and is only slightly tailored to the AMD Radeon Pro 5500M in my MacBook Pro.
 - Various gameplay optimizations from Ville[2].

## Results

TODO: add a table of results for different game sizes and algorithms, along with timings for CPU vs GPU.

## Docs

I'll be adding details on the tradeoffs in various parts of the implementation, the GPU algorithms, performance work,
etc. over time.

## Easier Implementations

I originally wrote similar code in Java, since I was first motivated by the APCS students. I've kept the Java version
pretty straightforward, and it is an easier introduction to the problem than the optimized C++ code you'll find here.
See https://github.com/mikemag/CS-Education/tree/master/APCS/Mastermind

## License

Code is licensed under the MIT license except as otherwise noted.
See [LICENSE](https://github.com/mikemag/Mastermind/blob/master/LICENSE) for details.

Documents and images are copyright by [Michael Magruder](https://github.com/mikemag) and licensed under a 
[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## References

[1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.

[2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March 2013, arXiv:1305.1010 [cs.GT]. https://arxiv.org/abs/1305.1010

[3] Barteld Kooi, Yet another mastermind strategy. International Computer Games Association Journal, 28(1):13–20, 2005. https://www.researchgate.net/publication/30485793_Yet_another_Mastermind_strategy

[4] Stuart Reges and Marty Stepp, Building Java Programs: a back to basics approach, 2nd edition, 2011, Addison-Wesley. https://www.buildingjavaprograms.com/
  