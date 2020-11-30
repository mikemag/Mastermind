// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <unordered_map>

#include "gpu_interface_wrapper.hpp"
#include "strategy.hpp"

// These strategies all rely on splitting the remaining possible guesses into groups or subsets based on their scores
// vs each other.
//
// References:
// [1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.
//
// [2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy<pinCount,c> and More Results in the Expected Case, March
// 2013, arXiv:1305.1010 [cs.GT].
//
// [3] Barteld Kooi, Yet another mastermind Strategy<pinCount,c>. International Computer Games Association Journal,
// 28(1):13–20, 2005. https://www.researchgate.net/publication/30485793_Yet_another_Mastermind_strategy

// --------------------------------------------------------------------------------------------------------------------
// Base for all subsetting strategies, CPU-based
//
// The core of this comes from the method described by Knuth in [1]. This subsets the possibilities by score, comparing
// all remaining codewords (i.e., not yet guessed) to all current possible guesses. This is O(n^2) in the number of
// total codewords, though the size of the possible solutions set does decrease drastically with each play. Of course, n
// = c^pinCount, so this is very, very expensive for larger games. Only the sizes of the subsets are retained; the
// actual codewords in each subset are unnecessary.
//
// Given these subset sizes, other algorithms can proceed to come up with a score for each possible guess, and the guess
// with the maximum score will be used, favoring guesses which are still in the possible solution set on ties.
//
// There's a decent summary of Knuth's overall algorithm on Wikipedia, too:
// https://en.wikipedia.org/wiki/Mastermind_(board_game)

template <uint8_t p, uint8_t c, bool l>
class StrategySubsetting : public Strategy<p, c, l> {
 public:
  StrategySubsetting() : Strategy<p, c, l>{} {
    for (auto &s : subsetSizes) {
      s = 0;
    }
  }

  explicit StrategySubsetting(Codeword<p, c> initialGuess) : Strategy<p, c, l>{initialGuess} {
    for (auto &s : subsetSizes) {
      s = 0;
    }
  }

  StrategySubsetting(Codeword<p, c> nextGuess, std::vector<Codeword<p, c>> &nextPossibleSolutions,
                     std::vector<uint32_t> &nextUsedCodewords)
      : Strategy<p, c, l>(nextGuess, nextPossibleSolutions), savedUsedCodewords(std::move(nextUsedCodewords)) {}

  Codeword<p, c> selectNextGuess() override;
  virtual size_t computeSubsetScore() = 0;

 protected:
  // Adds a set of codewords we've already guessed so we can ignore them from the set of all codewords when looking
  // for possibly inconsistent moves. You'd imagine an unordered_set<> would be best here, but the list is so short
  // that it's actually slower than a simple vector w/ linear search.
  const std::vector<uint32_t> savedUsedCodewords;
  std::vector<uint32_t> usedCodewords;

  // Storage for the subset counts, kept static and reused for an easy way to have them zero'd for each use. Also flat
  // and sparse, but that's okay, it's faster to use it in the inner loop than using a 2D array.
  static inline int subsetSizes[Strategy<p, c, l>::maxScoreSlots];

 private:
  int computeTotalSubsets();
};

// --------------------------------------------------------------------------------------------------------------------
// Base for all subsetting strategies, CPU-based
//
// This is a base for all subsetting algorithms that want to take advantage of the GPU. The core GPU setup and access
// logic is the same, as is the fallback to CPU-only when necessary. The only real variable here is the compute kernel
// name.
//
// Strategies which derrive from this run the same algorithm as their CPU versions, with all scoring and subset counting
// dones on a GPU via Apple's Metal API. A single compute kernel is passed buffers with codewords and pre-computed color
// counts: one set for all codewords, and one set for the possible solutions. Each GPU core gets a single entry G from
// the all codewords set and scores it against all elements of the possible solutions set, forming the subsets. Then,
// the subset score for G is found and placed in the correct place in a return buffer, along with a bool to tell us if G
// is also a possible solution.
//
// The GPU processes the entire group, then it's a simple matter of a single pass over the list of subset scores and
// selecting the best guess as per usual.
//
// This is a massive speed increase for larger games. For example, using Knuth's algorithm:
//
//           CPU         GPU
// 5p8c:   17.4421s    2.0124s
// 6p6c:   36.2458s    3.3394s
//
// See the implementation of this class for more details on how buffers are arranged and transfered to the GPU.
//
// See compute_kernels.metal for the implementation on the GPU side.
enum GPUMode { Both, GPU, CPU };

template <uint8_t pinCount, uint8_t c, bool l>
class StrategySubsettingGPU : public StrategySubsetting<pinCount, c, l> {
 public:
  StrategySubsettingGPU(const char *kernelName, GPUMode mode = Both)
      : StrategySubsetting<pinCount, c, l>{}, mode(mode) {
    setupGPUInterface(kernelName);
  }

  StrategySubsettingGPU(Codeword<pinCount, c> initialGuess, const char *kernelName)
      : StrategySubsetting<pinCount, c, l>{initialGuess} {
    setupGPUInterface(kernelName);
  }

  StrategySubsettingGPU(Codeword<pinCount, c> nextGuess, std::vector<Codeword<pinCount, c>> &nextPossibleSolutions,
                        std::vector<uint32_t> &nextUsedCodewords)
      : StrategySubsetting<pinCount, c, l>(nextGuess, nextPossibleSolutions, nextUsedCodewords) {}

  Codeword<pinCount, c> selectNextGuess() override;

  void printStats(std::chrono::duration<float, std::milli> elapsedMS) override;

 protected:
  GPUMode mode = Both;

 private:
  static inline GPUInterfaceWrapper *gpuInterface = nullptr;
  static inline bool allCodewordsOnGPU = false;
  static void copyAllCodewordsToGPU();
  static inline uint64_t kernelsExecuted = 0;

  void setupGPUInterface(const char *kernelName) {
    if (mode != CPU) {
      if (gpuInterface == nullptr) {
        gpuInterface = new GPUInterfaceWrapper(pinCount, (uint)Codeword<pinCount, c>::totalCodewords, kernelName);
        copyAllCodewordsToGPU();
      }
      if (!gpuInterface->gpuAvailable()) {
        mode = CPU;
      }
    }
  }
};

// --------------------------------------------------------------------------------------------------------------------
// Knuth: this is the method described by Knuth in [1]. There's a decent summary of this on Wikipedia, too:
// https://en.wikipedia.org/wiki/Mastermind_(board_game)
//
// The core of Knuth's algorithm: find the remaining solution which will eliminate the most possibilities on the next
// round, favoring, but not requiring, any choice which may still be the final answer.
//
// The attraction with this method is that it gives an upper bound on the number of moves needed to win the game.
// For a 4p6c game, it's 5 moves with an average of 4.4761 moves over all possible games.
//
//       Start    Avg   Max
// 4p6c: 1122   4.4761   5
// 4p7c: 1234   4.8367   6
// 5p8c: 11234  5.6142   7
//
// These results match those presented in [2], Tables 3, 4, & 5.

template <uint8_t p, uint8_t c, bool l>
class StrategyKnuth : public StrategySubsettingGPU<p, c, l> {
 private:
  constexpr static auto kernelName = "findKnuthGuessKernel";

 public:
  StrategyKnuth(GPUMode mode = Both) : StrategySubsettingGPU<p, c, l>{kernelName, mode} {
    this->guess = Codeword<p, c>(presetInitialGuess());
  }

  explicit StrategyKnuth(Codeword<p, c> initialGuess) : StrategySubsettingGPU<p, c, l>{initialGuess, kernelName} {}

  StrategyKnuth(Codeword<p, c> nextGuess, std::vector<Codeword<p, c>> &nextPossibleSolutions,
                std::vector<uint32_t> &nextUsedCodewords)
      : StrategySubsettingGPU<p, c, l>(nextGuess, nextPossibleSolutions, nextUsedCodewords) {}

  std::string getName() const override { return "Knuth"; }

  size_t computeSubsetScore() override;
  std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) override;

  constexpr uint32_t presetInitialGuess() {
    switch (Strategy<p, c, l>::packedPinsAndColors) {
      case 0x46:
        return 0x1122;
      case 0x47:
        return 0x1234;
      case 0x58:
        return 0x11234;
      default:
        return Strategy<p, c, l>::genericInitialGuess;
    }
  }
};

// --------------------------------------------------------------------------------------------------------------------
// Most Parts tries to maximize the number of subsets that will be generated, regardless of their size. This algorithm
// is from Kooi[3] in 2005 and does very well with the expected average number of moves.
//
//       Start    Avg   Max
// 4p6c: 1123   4.3735   6
// 4p7c: 1123   4.7430   6
// 5p8c: 11223  5.5491   8
//
// These results match those presented in [2], Tables 3, 4, & 5 except for the 5p8c result: I get 8 moves max, he shows
// 9 in Table 5.

template <uint8_t p, uint8_t c, bool l>
class StrategyMostParts : public StrategySubsettingGPU<p, c, l> {
 private:
  constexpr static auto kernelName = "findMostPartsGuessKernel";

 public:
  StrategyMostParts(GPUMode mode = Both) : StrategySubsettingGPU<p, c, l>{kernelName, mode} {
    this->guess = Codeword<p, c>(presetInitialGuess());
  }

  explicit StrategyMostParts(Codeword<p, c> initialGuess) : StrategySubsettingGPU<p, c, l>{initialGuess, kernelName} {}

  StrategyMostParts(Codeword<p, c> nextGuess, std::vector<Codeword<p, c>> &nextPossibleSolutions,
                    std::vector<uint32_t> &nextUsedCodewords)
      : StrategySubsettingGPU<p, c, l>(nextGuess, nextPossibleSolutions, nextUsedCodewords) {}

  std::string getName() const override { return "Most Parts"; }

  size_t computeSubsetScore() override;
  std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) override;

  constexpr uint32_t presetInitialGuess() {
    switch (Strategy<p, c, l>::packedPinsAndColors) {
      case 0x46:
      case 0x47:
        return 0x1123;
      case 0x58:
        return 0x11223;
      default:
        return Strategy<p, c, l>::genericInitialGuess;
    }
  }
};

// --------------------------------------------------------------------------------------------------------------------
// Expected Size attempts to minimize the expected size of subsets. See [3] for a good explanation. Essentially, this is
// trying to figure out the number of remaining codewords in the expected case, rather than the worst case that Knuth
// uses.
//
//       Start    Avg   Max
// 4p6c: 1123   4.3935   6
// 4p7c: 1234   4.7530   6
// 5p8c: 11234  5.5038   7
//
// These results are close to those presented in [2], Tables 3, 4, & 5, but don't match perfectly. Expected Size,
// Entropy, and other algorithms that produce a floating point score are sensitive to floating point error. In
// particular, the core of any such scheme involves breaking ties on equal scores, but equality w/ floating point is
// inherently error prone. Detailed investigation in this area has shown that the small variances here are all due to FP
// error.
//
// NB: I've chosen to round floating point scores to 16 decimal places.
//
// Note: Metal doesn't support doubles, so there's less precision in the expected size calc and it produces a slightly
// different result for each run. 4.3951 for 4p6c for instance.

template <uint8_t p, uint8_t c, bool l>
class StrategyExpectedSize : public StrategySubsettingGPU<p, c, l> {
 private:
  constexpr static auto kernelName = "findExpectedSizeGuessKernel";

 public:
  StrategyExpectedSize(GPUMode mode = Both) : StrategySubsettingGPU<p, c, l>{kernelName, mode} {
    this->guess = Codeword<p, c>(presetInitialGuess());
  }

  explicit StrategyExpectedSize(Codeword<p, c> initialGuess)
      : StrategySubsettingGPU<p, c, l>{initialGuess, kernelName} {}

  StrategyExpectedSize(Codeword<p, c> nextGuess, std::vector<Codeword<p, c>> &nextPossibleSolutions,
                       std::vector<uint32_t> &nextUsedCodewords)
      : StrategySubsettingGPU<p, c, l>(nextGuess, nextPossibleSolutions, nextUsedCodewords) {}

  std::string getName() const override { return "Expected Size"; }

  size_t computeSubsetScore() override;
  std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) override;

  constexpr uint32_t presetInitialGuess() {
    switch (Strategy<p, c, l>::packedPinsAndColors) {
      case 0x46:
        return 0x1123;
      case 0x47:
        return 0x1234;
      case 0x58:
        return 0x11234;
      default:
        return Strategy<p, c, l>::genericInitialGuess;
    }
  }
};

// --------------------------------------------------------------------------------------------------------------------
// Entropy uses the entropy of the subsets to pick the next guess, favoring the set with the highest entropy. See
// section 2.5 of [3] for a good explanation of the algorithm.
//
//       Start    Avg   Max
// 4p6c: 1234   4.4159   6
// 4p7c: 1234   4.7397   6
// 5p8c: 11234  5.4898   8
//
// These results match those presented in [2], Tables 3, 4, & 5. NB, I've chosen to round floating point scores to 16
// decimal places.
//
// Note: Metal doesn't support doubles, so there's less precision in the entropy calc and it produces a slightly
// different result for each run. 4.4151 for 4p6c for instance.

template <uint8_t p, uint8_t c, bool l>
class StrategyEntropy : public StrategySubsettingGPU<p, c, l> {
 private:
  constexpr static auto kernelName = "findEntropyGuessKernel";

 public:
  StrategyEntropy(GPUMode mode = Both) : StrategySubsettingGPU<p, c, l>{kernelName, mode} {
    this->guess = Codeword<p, c>(presetInitialGuess());
  }

  explicit StrategyEntropy(Codeword<p, c> initialGuess) : StrategySubsettingGPU<p, c, l>{initialGuess, kernelName} {}

  StrategyEntropy(Codeword<p, c> nextGuess, std::vector<Codeword<p, c>> &nextPossibleSolutions,
                  std::vector<uint32_t> &nextUsedCodewords)
      : StrategySubsettingGPU<p, c, l>(nextGuess, nextPossibleSolutions, nextUsedCodewords) {}

  std::string getName() const override { return "Entropy"; }

  size_t computeSubsetScore() override;
  std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) override;

  constexpr uint32_t presetInitialGuess() {
    switch (p) {
      case 4:
        return 0x1234;
      case 5:
        return 0x11234;
      default:
        return Strategy<p, c, l>::genericInitialGuess;
    }
  }
};

#include "subsetting_strategies.cpp"
