// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "strategy.hpp"

// All of these are very simple gameplay strategies which are very fast and easy to understand.

// --------------------------------------------------------------------------------------------------------------------
// First One: simply pick the next consistent guess in the list and play it.
//
// Since we're starting with a list, in order, of all possible guesses, and since we remove all inconsistent guesses
// from that list in each round, we can just select any one of them to make a good play. The first one is as good as
// any other, and it's fast, so that's what this does.
//
//       Start    Avg   Max
// 4p6c: 1122   5.0216   8
// 4p6c: 3456   4.6644   7
// 4p7c: 4567   5.1083   8
// 5p8c: 45678  5.9702   10
//
// These results match those presented in [2], Tables 3, 4, & 5.

template <uint8_t p, uint8_t c, bool l>
class StrategyFirstOne : public Strategy<p, c, l> {
 public:
  StrategyFirstOne() : Strategy<p, c, l>{} { this->guess = Codeword<p, c>(presetInitialGuess()); }
  explicit StrategyFirstOne(Codeword<p, c> initialGuess) : Strategy<p, c, l>{initialGuess} {}

  StrategyFirstOne(Strategy<p, c, l> &parent, Codeword<p, c> nextGuess,
                   std::vector<Codeword<p, c>> &nextPossibleSolutions)
      : Strategy<p, c, l>(parent, nextGuess, nextPossibleSolutions) {}

  std::string getName() const override { return "First One"; }

  Codeword<p, c> selectNextGuess() override;
  std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) override;

  constexpr uint32_t presetInitialGuess() {
    switch (Strategy<p, c, l>::packedPinsAndColors) {
      case 0x46:
        return 0x3456;
      case 0x47:
        return 0x4567;
      case 0x58:
        return 0x45678;
      default:
        return Strategy<p, c, l>::genericInitialGuess;
    }
  }
};

// --------------------------------------------------------------------------------------------------------------------
// Random: simply pick any of the remaining consistent guesses in the list and play it.
//
// Much like First One, but this picks a random guess from the list. This is surprisingly effective in producing
// quick wins, with very little effort.

template <uint8_t p, uint8_t c, bool l>
class StrategyRandom : public Strategy<p, c, l> {
 public:
  StrategyRandom() : Strategy<p, c, l>{} {}
  explicit StrategyRandom(Codeword<p, c> initialGuess) : Strategy<p, c, l>{initialGuess} {}

  StrategyRandom(Strategy<p, c, l> &parent, Codeword<p, c> nextGuess,
                 std::vector<Codeword<p, c>> &nextPossibleSolutions)
      : Strategy<p, c, l>(parent, nextGuess, nextPossibleSolutions) {}

  std::string getName() const override { return "Random"; }

  Codeword<p, c> selectNextGuess() override;
  std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) override;
};

#include "simple_strategies.cpp"
