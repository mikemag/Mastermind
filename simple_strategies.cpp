// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <random>

using namespace std;

// --------------------------------------------------------------------------------------------------------------------
// First One

template <uint8_t p, uint8_t c, bool log>
Codeword<p, c> StrategyFirstOne<p, c, log>::selectNextGuess() {
  auto &allCodewords = Codeword<p, c>::getAllCodewords();
  Codeword<p, c> nextGuess = allCodewords[this->possibleSolutions.front()];
  this->possibleSolutions.erase(this->possibleSolutions.begin());
  if (log) {
    cout << "Selecting the first possibility blindly: " << nextGuess << endl;
  }
  return nextGuess;
}

template <uint8_t p, uint8_t c, bool l>
shared_ptr<Strategy<p, c, l>> StrategyFirstOne<p, c, l>::createNewMove(Score r, Codeword<p, c> nextGuess) {
  auto next = make_shared<StrategyFirstOne<p, c, l>>(*this, nextGuess, this->possibleSolutions);
  return next;
}

// --------------------------------------------------------------------------------------------------------------------
// Random

// A good random number generator for Algo::Random
static std::random_device randDevice;
static std::mt19937 randGenerator(randDevice());

template <uint8_t p, uint8_t c, bool log>
Codeword<p, c> StrategyRandom<p, c, log>::selectNextGuess() {
  std::uniform_int_distribution<> distrib(0, (int)this->possibleSolutions.size() - 1);
  auto &allCodewords = Codeword<p, c>::getAllCodewords();
  uint32_t nextGuessIndex = this->possibleSolutions[distrib(randGenerator)];
  Codeword<p, c> nextGuess = allCodewords[nextGuessIndex];
  this->possibleSolutions.erase(remove(this->possibleSolutions.begin(), this->possibleSolutions.end(), nextGuessIndex),
                                this->possibleSolutions.end());
  if (log) {
    cout << "Selecting a random possibility: " << nextGuess << endl;
  }
  return nextGuess;
}

template <uint8_t p, uint8_t c, bool l>
shared_ptr<Strategy<p, c, l>> StrategyRandom<p, c, l>::createNewMove(Score r, Codeword<p, c> nextGuess) {
  auto next = make_shared<StrategyRandom<p, c, l>>(*this, nextGuess, this->possibleSolutions);
  return next;
}
