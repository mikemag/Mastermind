// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <set>

#include "codeword.hpp"
#include "compute_kernel_constants.h"
#include "mastermind_config.h"
#include "preset_initial_guesses.h"
#include "score.hpp"
#include "simple_strategies.hpp"
#include "strategy.hpp"
#include "subsetting_strategies.hpp"

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
static constexpr Algo singleGameAlgo = Algo::MostParts;
static constexpr GPUMode singleGameGPUMode = Both;
static constexpr uint8_t singleGamePinCount = 8;    // 1-8, 4 is classic
static constexpr uint8_t singleGameColorCount = 5;  // 1-15, 6 is classic

static constexpr bool playMultipleGames = false;     // Play a set of games defined below.
static constexpr bool runTests = false;               // Run unit tests and play Knuth's game
static constexpr bool findBestFirstGuesses = false;  // Initial guess exploration
static constexpr bool writeStratFiles = false;

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
  StrategyKnuth<4, 6, true> s(Codeword<4, 6>(0x1122), singleGameGPUMode);
  s.findSecret(Codeword<4, 6>(0x3632));
  printf("\n");
}

static constexpr uint histogramSize = 32;
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

  if (writeStratFiles) {
    strategy->dump();
  }

  cout << "Done" << endl;
}

template <uint8_t pinCount, uint8_t colorCount, bool log>
shared_ptr<Strategy<pinCount, colorCount, log>> makeStrategy(Algo algorithm, GPUMode mode) {
  switch (algorithm) {
    case FirstOne:
      return make_shared<StrategyFirstOne<pinCount, colorCount, log>>(
          presetInitialGuessFirstOne<pinCount, colorCount>());
    case Random:
      return make_shared<StrategyRandom<pinCount, colorCount, log>>(Codeword<pinCount, colorCount>::onePins);
    case Knuth:
      return make_shared<StrategyKnuth<pinCount, colorCount, log>>(presetInitialGuessKnuth<pinCount, colorCount>(),
                                                                   mode);
    case MostParts:
      return make_shared<StrategyMostParts<pinCount, colorCount, log>>(
          presetInitialGuessMostParts<pinCount, colorCount>(), mode);
    case ExpectedSize:
      return make_shared<StrategyExpectedSize<pinCount, colorCount, log>>(
          presetInitialGuessExpectedSize<pinCount, colorCount>(), mode);
    case Entropy:
      return make_shared<StrategyEntropy<pinCount, colorCount, log>>(presetInitialGuessEntropy<pinCount, colorCount>(),
                                                                     mode);
    default:
      return nullptr;
  }
};

template <uint8_t pinCount, uint8_t colorCount, bool log>
shared_ptr<Strategy<pinCount, colorCount, log>> makeStrategy(Algo algorithm, uint32_t initialGuessPacked,
                                                             GPUMode mode) {
  Codeword<pinCount, colorCount> initialGuess(initialGuessPacked);
  switch (algorithm) {
    case FirstOne:
      return make_shared<StrategyFirstOne<pinCount, colorCount, log>>(initialGuess);
    case Random:
      return make_shared<StrategyRandom<pinCount, colorCount, log>>(initialGuess);
    case Knuth:
      return make_shared<StrategyKnuth<pinCount, colorCount, log>>(initialGuess, mode);
    case MostParts:
      return make_shared<StrategyMostParts<pinCount, colorCount, log>>(initialGuess, mode);
    case ExpectedSize:
      return make_shared<StrategyExpectedSize<pinCount, colorCount, log>>(initialGuess, mode);
    case Entropy:
      return make_shared<StrategyEntropy<pinCount, colorCount, log>>(initialGuess, mode);
    default:
      return nullptr;
  }
};

// Find a unique set of initial guesses. Using different digits for the same pattern isn't useful, nor are shuffled
// patterns. For 4p6c the unique initial guesses ae 1111, 1112, 1122, 1123, 1234. Repetitions of the same pattern, such
// as 2222, 2111, 3456, 1223, etc. aren't useful as they yield the same information.
//
// This is pretty brute-force. I feel like I'm missing a clever algorithm or representation for these.
template <uint8_t p, uint8_t c>
set<uint32_t> buildInitialGuessSet(Algo algo) {
  set<uint32_t> initialGuessSet;
  auto& allCodewords = Codeword<p, c>::getAllCodewords();
  for (auto& codeword : allCodewords) {
    auto packed = codeword.packedCodeword();

    // Count the colors.
    vector<uint8_t> colorCounts(16, 0);
    while (packed != 0) {
      uint8_t d = packed & 0xFu;
      colorCounts[d]++;
      packed >>= 4u;
    }

    // Sort the counts in reverse. Gives us the pattern we'll use next.
    sort(begin(colorCounts), end(colorCounts), greater<>());

    if (colorCounts[0] == p) {
      // Never a good choice, so just skip it even though it is valid.
      continue;
    }

    // Form up a new codeword using the pattern in colorCounts.
    uint32_t uniquePacked = 0;
    uint8_t nextDigit = 1;
    if (algo == FirstOne) {
      nextDigit = c;  // Use the highest numbers for First One, to trim off the early guesses
    }
    for (auto count : colorCounts) {
      for (int i = 0; i < count; i++) {
        uniquePacked = (uniquePacked << 4u) | nextDigit;
      }
      if (algo == FirstOne) {
        nextDigit--;
      } else {
        nextDigit++;
      }
    }

    // Toss it in a set to uniquify them
    initialGuessSet.insert(uniquePacked);

    if (colorCounts[0] == 1) {
      // Once we hit all unique digits we can stop. Subsequent codewords are guaranteed to be dups.
      break;
    }
  }

  printf("Unique initial guesses: ");
  for (auto cp : initialGuessSet) {
    printf("%x, ", cp);
  }
  printf("\n\n");

  return initialGuessSet;
}

template <uint8_t p, uint8_t c>
void runWithAllInitialGuesses(Algo a, StatsRecorder& s) {
  set<uint32_t> igs = buildInitialGuessSet<p, c>(a);
  for (auto ig : igs) {
    s.newRun();
    playAllGamesForStrategy(makeStrategy<p, c, false>(a, ig, Both), s);
  }
}

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
  string csvTag;

  if (playSingleGame) {
    statsRecorder.newRun();
    auto strategy = makeStrategy<singleGamePinCount, singleGameColorCount, false>(singleGameAlgo, singleGameGPUMode);
    playAllGamesForStrategy(strategy, statsRecorder);
  }

  if (playMultipleGames) {
    // NB: the templating of all of this means that multiple copies of much of the code are created for each entry in
    // this table. So only create it if we're actually playing multiple games. This keeps build times lower during
    // development.
    csvTag = "6_7_8_med_";
    static vector<void (*)(Algo, GPUMode, StatsRecorder&)> games = {
        [](Algo a, GPUMode m, StatsRecorder& s) { playAllGamesForStrategy(makeStrategy<6, 9, false>(a, m), s); },
        [](Algo a, GPUMode m, StatsRecorder& s) { playAllGamesForStrategy(makeStrategy<6, 10, false>(a, m), s); },

        [](Algo a, GPUMode m, StatsRecorder& s) { playAllGamesForStrategy(makeStrategy<7, 7, false>(a, m), s); },

        [](Algo a, GPUMode m, StatsRecorder& s) { playAllGamesForStrategy(makeStrategy<8, 5, false>(a, m), s); },
        [](Algo a, GPUMode m, StatsRecorder& s) { playAllGamesForStrategy(makeStrategy<8, 6, false>(a, m), s); },
    };

    static vector<Algo> interestingAlgos = {Knuth, MostParts, Entropy, ExpectedSize, FirstOne};

#ifdef __MM_GPU_METAL__
    static vector<GPUMode> gpuModes = {CPU, Both};
#else
    static vector<GPUMode> gpuModes = {CPU};
#endif

    for (auto& f : games) {
      for (auto& a : interestingAlgos) {
        for (auto& m : gpuModes) {
          statsRecorder.newRun();
          f(a, m, statsRecorder);
        }
      }
    }
  }

  if (findBestFirstGuesses) {
    csvTag = "find_ig_first_one_8_";
    constexpr static uint8_t pc = 8;
    static vector<void (*)(Algo, StatsRecorder&)> games = {
        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 2>(a, s); },
        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 3>(a, s); },
        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 4>(a, s); },
        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 5>(a, s); },
        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 6>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 7>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 8>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 9>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 10>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 11>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 12>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 13>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 14>(a, s); },
        //        [](Algo a, StatsRecorder& s) { runWithAllInitialGuesses<pc, 15>(a, s); },
    };
    static vector<Algo> interestingAlgos = {Knuth, MostParts, Entropy, ExpectedSize, FirstOne};

    for (auto& f : games) {
      for (auto& a : interestingAlgos) {
        f(a, statsRecorder);
      }
    }
  }

  tm t = {};
  istringstream ss(MASTERMIND_GIT_COMMIT_DATE);
  ss >> get_time(&t, "%Y-%m-%d %H:%M:%S");
  stringstream fs;
  fs << "mastermind_run_stats_" << csvTag << put_time(&t, "%Y%m%d_%H%M%S") << "_" << MASTERMIND_GIT_COMMIT_HASH
     << ".csv";
  statsRecorder.writeStats(fs.str());

  return 0;
}

//Run the example from Knuth's paper to compare with his results.
//Starting search for secret 3632, initial guess is 1122 with 1,296 possibilities.
//
//Tried guess 1122 against secret 3632 => 10
//Removing inconsistent possibilities... 256 remain.
//Selecting best guess: 1344	score: 212
//
//Tried guess 1344 against secret 3632 => 01
//Removing inconsistent possibilities... 44 remain.
//Selecting best guess: 3526	score: 37
//
//Tried guess 3526 against secret 3632 => 12
//Removing inconsistent possibilities... 7 remain.
//Selecting fully discriminating guess: 1462, subsets: 7
//
//Tried guess 1462 against secret 3632 => 11
//Removing inconsistent possibilities... 1 remain.
//Only remaining solution must be correct: 3632
//
//Tried guess 3632 against secret 3632 => 40
//Solution found after 5 moves.
