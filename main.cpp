// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "codeword.hpp"
#include "score.hpp"
#include "simple_strategies.hpp"
#include "strategy.hpp"
#include "subsetting_strategies.hpp"
#include "utils.hpp"

using namespace std;

// Mastermind
//
// Play the game Mastermind, which is to guess a sequence of colors (the "secret") with feedback
// about how many you have guessed correctly, and how many are in the right place.
//
// This will play the game for every possible secret and tell us the average and maximum number of
// tries needed across all of them.
//
// There are a few algorithms to play with. See the various Strategy class implementations for details.

enum Algo {
  FirstOne,      // Pick the first of the remaining choices.
  Random,        // Pick any of the remaining choices.
  Knuth,         // Pick the one that will eliminate the most remaining choices.
  MostParts,     // Mazimize the number of scores at each round.
  ExpectedSize,  // Minimize the expected size of the remaining choices.
  Entropy,       // Pick the maximum entropy guess.
};

// Config for a single game
static constexpr bool playSingleGame = true;
static constexpr Algo algo = Algo::Knuth;
static constexpr uint8_t pinCount = 5;    // 1-8, 4 is classic
static constexpr uint8_t colorCount = 8;  // 1-15, 6 is classic

static constexpr bool playMultipleGames = false;  // Play a set of games defined below.
static constexpr bool runTests = true;            // Run unit tests and play Knuth's game

static constexpr GPUMode gpuMode = Both;

void runUnitTests() {
  // Test cases from Miyoshi
  constexpr Codeword<4, 6> testSecret(0x6684);
  bool success = true;
  success &= (testSecret.score(Codeword<4, 6>(0x0000)) == Score(0, 0));
  success &= (testSecret.score(Codeword<4, 6>(0x6666)) == Score(2, 0));
  success &= (testSecret.score(Codeword<4, 6>(0x0123)) == Score(0, 0));
  success &= (testSecret.score(Codeword<4, 6>(0x4567)) == Score(0, 2));
  success &= (testSecret.score(Codeword<4, 6>(0x4589)) == Score(1, 1));
  success &= (testSecret.score(Codeword<4, 6>(0x6700)) == Score(1, 0));
  success &= (testSecret.score(Codeword<4, 6>(0x0798)) == Score(0, 1));
  success &= (testSecret.score(Codeword<4, 6>(0x6484)) == Score(3, 0));
  success &= (testSecret.score(Codeword<4, 6>(0x6480)) == Score(2, 1));
  success &= (testSecret.score(Codeword<4, 6>(0x6884)) == Score(3, 0));
  success &= (testSecret.score(Codeword<4, 6>(0x6684)) == Score(4, 0));

  // Three extra tests to detect subtly broken scoring functions.
  success &= (testSecret.score(Codeword<4, 6>(0x8468)) == Score(0, 3));
  success &= (testSecret.score(Codeword<4, 6>(0x8866)) == Score(0, 3));
  success &= (testSecret.score(Codeword<4, 6>(0x8466)) == Score(0, 4));

  if (success) {
    printf("Tests pass\n");
  } else {
    printf("Some tests failed!\n");
    exit(-1);
  }
}

void runKnuthTest() {
  printf("\nRun the example from Knuth's paper to compare with his results.\n");
  StrategyKnuth<4, 6, true> s(Codeword<4, 6>(0x1122));
  s.findSecret(Codeword<4, 6>(0x3632));
  printf("\n");
}

template <uint8_t pinCount, uint8_t colorCount, bool log>
void playAllGamesForStrategy(shared_ptr<Strategy<pinCount, colorCount, log>> strategy) {
  auto &allCodewords = Codeword<pinCount, colorCount>::getAllCodewords();
  printf("Playing all %d pin %d color games using algorithm '%s' for every possible secret...\n", pinCount, colorCount,
         strategy->getName().c_str());
  cout << "Total codewords: " << commaString(allCodewords.size()) << endl;
  cout << "Initial guess: " << strategy->currentGuess() << endl;

  int maxTurns = 0;
  int totalTurns = 0;
  constexpr uint histogramSize = 16;
  vector<int> histogram(histogramSize, 0);
  Codeword<pinCount, colorCount> maxSecret;
  bool showProgress = true;
  int progressFactor = 1000;
  int gameCount = 0;
  auto startTime = chrono::high_resolution_clock::now();
  auto lastTime = startTime;

  for (const auto &secret : allCodewords) {
    uint turns = strategy->findSecret(secret);
    totalTurns += turns;
    histogram[min(turns, histogramSize)]++;
    if (turns > maxTurns) {
      maxTurns = turns;
      maxSecret = secret;
    }
    if (showProgress) {
      gameCount++;
      if (gameCount > 0 && gameCount % progressFactor == 0) {
        auto currentTime = chrono::high_resolution_clock::now();
        chrono::duration<float> elapsedS = currentTime - lastTime;
        if (gameCount == progressFactor && elapsedS.count() < 0.5f) {
          showProgress = false;
          continue;
        }
        float eta = (allCodewords.size() - gameCount) / 1000.0 * elapsedS.count();
        cout << "Completed " << secret;
        printf(", %.04fs per %d, %.02f%%, ETA %.02fs\n", elapsedS.count(), progressFactor,
               ((float)gameCount / allCodewords.size()) * 100.0f, eta);
        lastTime = currentTime;
      }
    }
  }

  auto endTime = chrono::high_resolution_clock::now();
  double averageTurns = (double)totalTurns / allCodewords.size();
  printf("Average number of turns was %.4f\n", averageTurns);
  cout << "Maximum number of turns over all possible secrets was " << maxTurns << " with secret " << maxSecret << endl;
  strategy->printScoreCounters();
  chrono::duration<float, milli> elapsedMS = endTime - startTime;
  cout << "Elapsed time " << commaString(elapsedMS.count() / 1000) << "s, average search "
       << commaString(elapsedMS.count() / allCodewords.size()) << "ms" << endl;
  strategy->printStats(elapsedMS);

  printf("\n");
  for (int i = 0; i < histogramSize; i++) {
    if (histogram[i] > 0) {
      printf("%2d: %s ", i, commaString(histogram[i]).c_str());
    }
  }
  printf("\n");

  strategy->dump();
  strategy->dumpExperimentStats();

  cout << "Done" << endl;
}

template <uint8_t pinCount, uint8_t colorCount, bool log>
shared_ptr<Strategy<pinCount, colorCount, log>> makeStrategyWithAlgo(Algo algorithm) {
  switch (algorithm) {
    case FirstOne:
      return make_shared<StrategyFirstOne<pinCount, colorCount, log>>();
    case Random:
      return make_shared<StrategyRandom<pinCount, colorCount, log>>();
    case Knuth:
      return make_shared<StrategyKnuth<pinCount, colorCount, log>>(gpuMode);
    case MostParts:
      return make_shared<StrategyMostParts<pinCount, colorCount, log>>(gpuMode);
    case ExpectedSize:
      return make_shared<StrategyExpectedSize<pinCount, colorCount, log>>(gpuMode);
    case Entropy:
      return make_shared<StrategyEntropy<pinCount, colorCount, log>>(gpuMode);
  }
};

// Specific games to play
// static vector<void (*)()> manyGamesSpecific = {
//    []() { playAllGamesForStrategy(makeStrategyWithAlgo<4, 6, false>(Algo::Knuth)); },
//    []() { playAllGamesForStrategy(makeStrategyWithAlgo<4, 7, false>(Algo::Knuth)); },
//    []() { playAllGamesForStrategy(makeStrategyWithAlgo<5, 7, false>(Algo::Knuth)); },
//    []() { playAllGamesForStrategy(makeStrategyWithAlgo<5, 8, false>(Algo::Knuth)); },
//};

int main(int argc, const char *argv[]) {
  if (runTests) {
    runUnitTests();
    runKnuthTest();
  }

  if (playSingleGame) {
    auto strategy = makeStrategyWithAlgo<pinCount, colorCount, false>(algo);
    playAllGamesForStrategy(strategy);
  }

  if (playMultipleGames) {
    // NB: the templating of all of this means that multiple copies of much of the code are created for each entry in
    // this table. So only create it if we're actually playinig multiple games. This keeps build times lower during
    // development.
    static vector<void (*)(Algo algorithm)> manyGamesByAlgo = {
        [](Algo algorithm) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 6, false>(algorithm)); },
        [](Algo algorithm) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 7, false>(algorithm)); },
        [](Algo algorithm) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 10, false>(algorithm)); },
        [](Algo algorithm) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 7, false>(algorithm)); },
        [](Algo algorithm) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 8, false>(algorithm)); },
    };

    static vector<Algo> interestingAlgos = {Knuth, MostParts, Entropy, ExpectedSize, FirstOne};

    for (auto &a : interestingAlgos) {
      for (auto &f : manyGamesByAlgo) {
        f(a);
      }
    }
  }

  return 0;
}
