// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "codeword.hpp"
#include "compute_kernel_constants.h"
#include "mastermind_config.h"
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

static constexpr uint histogramSize = 16;
static vector<int> histogram(histogramSize, 0);
static vector<string> histogramHeaders;

static void setupHistogramHeaders() {
  for (int i = 0; i < histogramSize; i++) {
    std::stringstream ss;
    ss << "Hist_" << setw(2) << setfill('0') << i;
    histogramHeaders.emplace_back(ss.str());
  }
}

template <uint8_t pinCount, uint8_t colorCount, bool log>
void playAllGamesForStrategy(shared_ptr<Strategy<pinCount, colorCount, log>> strategy, StatsRecorder& statsRecorder) {
  auto& allCodewords = Codeword<pinCount, colorCount>::getAllCodewords();
  printf("Playing all %d pin %d color games using algorithm '%s' for every possible secret...\n", pinCount, colorCount,
         strategy->getName().c_str());
  statsRecorder.add("Pin Count", (int)pinCount);
  statsRecorder.add("Color Count", (int)colorCount);
  statsRecorder.add("Strategy", strategy->getName());

  cout << "Total codewords: " << commaString(allCodewords.size()) << endl;
  statsRecorder.add("Total Codewords", allCodewords.size());

  Codeword<pinCount, colorCount> initialGuess = strategy->currentGuess();
  cout << "Initial guess: " << strategy->currentGuess() << endl;
  statsRecorder.add("Initial Guess", initialGuess);

  int maxTurns = 0;
  int totalTurns = 0;
  fill(begin(histogram), end(histogram), 0);
  Codeword<pinCount, colorCount> maxSecret;
  bool showProgress = true;
  int progressFactor = 1000;
  int gameCount = 0;
  auto startTime = chrono::high_resolution_clock::now();
  auto lastTime = startTime;

  for (const auto& secret : allCodewords) {
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
  statsRecorder.add("Average Turns", averageTurns);
  cout << "Maximum number of turns over all possible secrets was " << maxTurns << " with secret " << maxSecret << endl;
  statsRecorder.add("Max Turns", maxTurns);
  statsRecorder.add("Max Secret", maxSecret);
  chrono::duration<float, milli> elapsedMS = endTime - startTime;
  auto elapsedS = elapsedMS.count() / 1000;
  auto avgGameTimeMS = elapsedMS.count() / allCodewords.size();
  cout << "Elapsed time " << commaString(elapsedS) << "s, average search " << commaString(avgGameTimeMS) << "ms"
       << endl;
  statsRecorder.add("Elapsed (s)", elapsedS);
  statsRecorder.add("Average Game Time (ms)", avgGameTimeMS);
  strategy->printStats(elapsedMS);
  strategy->recordStats(statsRecorder, elapsedMS);

  printf("\n");
  for (int i = 0; i < histogramSize; i++) {
    if (histogram[i] > 0) {
      printf("%2d: %s ", i, commaString(histogram[i]).c_str());
    }
    statsRecorder.add(histogramHeaders[i], histogram[i]);
  }
  printf("\n");

  strategy->dump();

  cout << "Done" << endl;
}

template <uint8_t pinCount, uint8_t colorCount, bool log>
shared_ptr<Strategy<pinCount, colorCount, log>> makeStrategyWithAlgo(Algo algorithm, GPUMode mode = gpuMode) {
  switch (algorithm) {
    case FirstOne:
      return make_shared<StrategyFirstOne<pinCount, colorCount, log>>();
    case Random:
      return make_shared<StrategyRandom<pinCount, colorCount, log>>();
    case Knuth:
      return make_shared<StrategyKnuth<pinCount, colorCount, log>>(mode);
    case MostParts:
      return make_shared<StrategyMostParts<pinCount, colorCount, log>>(mode);
    case ExpectedSize:
      return make_shared<StrategyExpectedSize<pinCount, colorCount, log>>(mode);
    case Entropy:
      return make_shared<StrategyEntropy<pinCount, colorCount, log>>(mode);
  }
};

int main(int argc, const char* argv[]) {
  setupHistogramHeaders();

  if (runTests) {
    runUnitTests();
    runKnuthTest();
  }

  StatsRecorder statsRecorder;
  statsRecorder.addAll("Git Branch", MASTERMIND_GIT_BRANCH);
  statsRecorder.addAll("Git Commit Hash", MASTERMIND_GIT_COMMIT_HASH);
  statsRecorder.addAll("Git Commit Date", MASTERMIND_GIT_COMMIT_DATE);

  if (playSingleGame) {
    statsRecorder.newRun();
    auto strategy = makeStrategyWithAlgo<pinCount, colorCount, false>(algo);
    playAllGamesForStrategy(strategy, statsRecorder);
  }

  if (playMultipleGames) {
    // NB: the templating of all of this means that multiple copies of much of the code are created for each entry in
    // this table. So only create it if we're actually playinig multiple games. This keeps build times lower during
    // development.
    static vector<void (*)(Algo, StatsRecorder&)> manyGamesByAlgo = {
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<2, 15, false>(a), s); },

        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<3, 15, false>(a), s); },

        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 15, false>(a), s); },

        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 15, false>(a), s); },

        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<6, 15, false>(a), s); },

        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<7, 15, false>(a), s); },

        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 2, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 3, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 4, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 5, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 6, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 7, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 8, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 9, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 10, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 11, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 12, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 13, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 14, false>(a), s); },
        //        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<8, 15, false>(a), s); },
    };

    static vector<void (*)(Algo, StatsRecorder&)> manyCommonGames = {
        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 6, false>(a, CPU), s); },
        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 7, false>(a, CPU), s); },
        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 8, false>(a, CPU), s); },
#ifdef __MM_GPU_METAL__
        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 6, false>(a, GPU), s); },
        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<4, 7, false>(a, GPU), s); },
        [](Algo a, StatsRecorder& s) { playAllGamesForStrategy(makeStrategyWithAlgo<5, 8, false>(a, GPU), s); },
#endif
    };

    static vector<Algo> interestingAlgos = {Knuth, MostParts, Entropy, ExpectedSize, FirstOne};

    for (auto& f : manyCommonGames) {
      for (auto& a : interestingAlgos) {
        statsRecorder.newRun();
        f(a, statsRecorder);
      }
    }
  }

  tm t = {};
  istringstream ss(MASTERMIND_GIT_COMMIT_DATE);
  ss >> get_time(&t, "%Y-%m-%d %H:%M:%S");
  stringstream fs;
  fs << "mastermind_run_stats_" << put_time(&t, "%Y%m%d_%H%M%S") << "_" << MASTERMIND_GIT_COMMIT_HASH << ".csv";
  statsRecorder.writeStats(fs.str());

  return 0;
}
