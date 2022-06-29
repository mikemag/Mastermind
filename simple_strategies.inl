// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <random>

using namespace std;

// --------------------------------------------------------------------------------------------------------------------
// First One

template <typename StrategyConfig>
typename Strategy<StrategyConfig>::CodewordT StrategyFirstOne<StrategyConfig>::selectNextGuess() {
  CodewordT nextGuess = this->possibleSolutions.front();
  this->possibleSolutions.erase(this->possibleSolutions.begin());
  if (StrategyConfig::LOG) {
    cout << "Selecting the first possibility blindly: " << nextGuess << endl;
  }
  return nextGuess;
}

template <typename StrategyConfig>
shared_ptr<Strategy<StrategyConfig>> StrategyFirstOne<StrategyConfig>::createNewMove(Score r, CodewordT nextGuess) {
  auto next = make_shared<StrategyFirstOne<StrategyConfig>>(*this, nextGuess, this->possibleSolutions);
  return next;
}

// --------------------------------------------------------------------------------------------------------------------
// Random

// A good random number generator for Algo::Random
static std::random_device randDevice;
static std::mt19937 randGenerator(randDevice());

template <typename StrategyConfig>
typename Strategy<StrategyConfig>::CodewordT StrategyRandom<StrategyConfig>::selectNextGuess() {
  std::uniform_int_distribution<> distrib(0, (int)this->possibleSolutions.size() - 1);
  CodewordT nextGuess = this->possibleSolutions[distrib(randGenerator)];
  this->possibleSolutions.erase(remove(this->possibleSolutions.begin(), this->possibleSolutions.end(), nextGuess),
                                this->possibleSolutions.end());
  if (StrategyConfig::LOG) {
    cout << "Selecting a random possibility: " << nextGuess << endl;
  }
  return nextGuess;
}

template <typename StrategyConfig>
shared_ptr<Strategy<StrategyConfig>> StrategyRandom<StrategyConfig>::createNewMove(Score r, CodewordT nextGuess) {
  auto next = make_shared<StrategyRandom<StrategyConfig>>(*this, nextGuess, this->possibleSolutions);
  return next;
}
