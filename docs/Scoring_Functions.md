# Efficient Scoring Functions

This implementation has four variations on the scoring function. The original, somewhat obvious version (though still a bit optimized)
consumes roughly 4.4948s for all scores necessary to apply Knuth's algorithm to a $4p6c$ game. The most optimized version
here requires just 0.9237s.

I'm not going to describe the details of each variation. There are comments above each one in [codeword.cpp](../codeword.cpp).
Instead, I'll just describe the final one here.

## Scoring

Mastermind guesses are evaluated against the secret and given a score made up of a number of black $b$ and white $w$ hits.
Black hits are awarded for every correct color in the correct position, and white hits are awarded for every correct
color in the wrong position. Each colored pin in the secret can be used only once. Scores are typically represented
by the two numbers $b$ and $w$, either as separate values or packed together into a single value.
Games have a number of pins $p$ and colors $c$.

This can be a little tricky to understand and get right the first time. Knuth describes the scoring function well in [1].

Scoring two codewords is central to every algorithm for playing Mastermind. It is done millions, billions, even trillions
of times when playing all games of larger sizes. It's important that it is fast.

## scoreCountingHandVec()

The current implementation is in [codeword.cpp](../codeword.cpp). I'll copy it here for reference. For codegen, see
it on [Compiler Explorer](https://godbolt.org/z/KvPf1Y), it's only 21 instructions.

````c++
template <uint8_t pinCount, uint8_t c>
Score Codeword<pinCount, c>::scoreCountingHandVec(const Codeword &guess) const {
  constexpr static uint32_t unusedPinsMask = (uint32_t)(0xFFFFFFFFlu << (pinCount * 4u));
  uint32_t v = this->codeword ^ guess.codeword;  // Matched pins are now 0.
  v |= unusedPinsMask;                           // Ensure that any unused pin positions are non-zero.
  uint32_t r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint8_t b = std::popcount(r);

  // Load the 128-bit color counts into vector registers. Each one is 16 8-bit counters.
  __m128i_u secretColorsVec = _mm_loadu_si128((__m128i_u *)&this->colorCounts8);
  __m128i_u guessColorsVec = _mm_loadu_si128((__m128i_u *)&guess.colorCounts8);

  // Find the minimum of each pair of 8-bit counters in one instruction.
  __m128i_u minColorsVec = _mm_min_epu8(secretColorsVec, guessColorsVec);

  // Add up all of the 8-bit counters into two 16-bit sums in one instruction.
  __m128i vsum = _mm_sad_epu8(minColorsVec, _mm_setzero_si128());

  // Pull out the two 16-bit sums and add them together normally to get our final answer. 3 instructions.
  int allHits = _mm_extract_epi16(vsum, 0) + _mm_extract_epi16(vsum, 4);

  return Score(b, allHits - b);
}
````

### Codeword Representation

Codewords can be 2 to 8 pins, each represented by 4 bits and packed into a single 32 bit word. The scoring function
is given the number of pins as a compile-time constant via the template parameter `pinCount`.

### Pre-computed Color Counts

The count of each pin color is constant and pre-computed and stored alongside the codeword. This allows us to quickly use them each
time a codeword is scored. For this variation of the function, the counts are each 8 bits and 16 are packed into a 128 bit value.

### Computing B in Constant Time

To compute the number of black hits $b$ we simply need to count each place where pairs of pins in the same position match.
We can take a quick step towards getting our answer using xor: every nibble where pins match will be 0, and every nibble
were they were different will be non-zero.

Now we just need to count the zero nibbles. Building on the algorithm for determining if a word has a zero byte on
[Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord)[2], we form a single one bit for
every zero nibble and thus every matched pin as follows, acting on every nibble at the same time:

* zero the high bit in the nibble
* add 7 to the nibble. If it was 0, it will be 7 now and the high bit will still be clear. If it was anything else, it will 
be $\ge 8$ now and thus the high bit will be set.
* put the original high bits back. Now every nibble that was non-zero has the high bit set.
* set the three low bits in every nibble. Now ever nibble that was non-zero has all bits set, and every nibble which was zero has only the high bit clear.
* invert the result: we now have a single bit set for every nibble which was zero, i.e., every matched pin between the codewords.

All that is left is to count those bits. This can be done in a single `popcnt` instruction, 
available since the Nehalem architecture on Intel processors (2008) and Barcelona on AMD processor (2007).

This may seem complex, but it's not bad once you get your head around it the first time. If you count the operations vs.
other methods you'll find that this is actually pretty short. A nice benefit is that it's all operating on registers 
or immediates. Each op is quite fast, and they schedule well with other work.
  
## Hand Vectorization FTW

Computing $w$ in constant time starts with compuing the total of all hits, $b$ and $w$, by using the color counts.
Taking the min of each color count between the codewords, then summing them gives us all hits. We can then just subtract
$b$ to get $w$.

This is done by loading the 128 bit groups of color counts into vector registers, then applying a parallel minimum 
operation across all pairs of bytes with `_mm_min_epu8`.
This is a SSE2 instruction, `pminub`, with a latency of 1 and a CPI of 0.5, believe it or not.

Now we have 16 8-bit values which we just need to sum. This starts with a horizontal add via `_mm_sad_epu8` (`psadbw`) which yields
two 16-bit values, each containing sums for half the values. Finally, those two 16 bit values are extracted back into
scalar registers, and added with a regular scalar `add` instruction.

This ends up being roughly 5x faster than the original C++ version here. And it ports well to the GPU. 
See [Mastermind on the GPU](Mastermind_on_the_GPU.md) for details.

## References

[1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.

[2] Sean Eron Anderson, Bit Twiddling Hacks, https://graphics.stanford.edu/~seander/bithacks.html, retrieved December 2020.
