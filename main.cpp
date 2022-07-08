// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "codeword.hpp"
#include "mastermind_config.h"
#include "score.hpp"
#include "solver.hpp"
#include "solver_cpu_reference.hpp"
#include "solver_sets.hpp"

#ifdef __CUDACC__
#include "solver_cuda.hpp"
#endif

using namespace std;
namespace ss = solver_sets;

// Mastermind
//
// Play the game Mastermind, which is to guess a sequence of colors (the "secret") with feedback
// about how many you have guessed correctly, and how many are in the right place.
//
// This will play the game for every possible secret and tell us the average and maximum number of
// tries needed across all of them.
//
// There are a few algorithms to play with. See algos.hpp.

#ifdef __CUDACC__
template <typename T>
using DefaultSolver = SolverCUDA<T>;
#else
template <typename T>
using DefaultSolver = SolverReferenceImpl<T>;
#endif

// Config for a single game
static constexpr bool shouldPlaySingleGame = true;
template <typename T>
using SingleGameSolver = DefaultSolver<T>;
using SingleGameAlgo = Algos::Knuth;
static constexpr uint8_t singleGamePinCount = 8;    // 1-8, 4 is classic
static constexpr uint8_t singleGameColorCount = 5;  // 1-15, 6 is classic
static constexpr bool singleGameLog = true;

// Config for playing a set of games
static constexpr bool shouldPlayMultipleGames = true;
template <typename T>
using MultiGameSolver = SolverCUDA<T>;
using MultiGameAlgos = ss::algo_list<Algos::Knuth, Algos::MostParts>;
using MultiGamePins = ss::pin_counts<4, 5>;
using MultiGameColors = ss::color_counts<6>;
static constexpr bool multiGameLog = true;

// Initial guess exploration, plays the same games as the multi game config above
static constexpr bool shouldFindBestFirstGuesses = true;

// Misc config
static constexpr bool shouldRunTests = true;  // Run unit tests and play Knuth's game
static constexpr bool shouldWriteStratFiles = false;

template <bool shouldRun>
void runUnitTests() {
  if constexpr (shouldRun) {
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
}

// mmmfixme: bring this back and post-process the results to print the data we need to match the paper, or add an option
// to dump a single game as it runs.
#if 0
void runKnuthTest() {
  printf("\nRun the example from Knuth's paper to compare with his results.\n");
  StrategyKnuth<StrategyConfig<4, 6, true>> s(Codeword<4, 6>(0x1122));
  Codeword<4, 6> secret(0x3632);
  s.findSecret(secret);
  printf("\n");
}
#endif

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

template <typename Solver>
void runSingleSolver(StatsRecorder& statsRecorder, uint32_t packedInitialGuess) {
  using SolverConfig = typename Solver::SolverConfig;
  using CodewordT = typename SolverConfig::CodewordT;

  Solver solver;

  printf("Playing all %d pin %d color games using algorithm '%s' for every possible secret...\n",
         SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name);
  statsRecorder.newRun();
  statsRecorder.add("Pin Count", (int)SolverConfig::PIN_COUNT);
  statsRecorder.add("Color Count", (int)SolverConfig::COLOR_COUNT);
  statsRecorder.add("Strategy", SolverConfig::ALGO::name);
  // mmmfixme: get solver name working. Would be nice to let the
  //  Solver add whatever it wants to the sr.

  cout << "Total codewords: " << commaString(CodewordT::TOTAL_CODEWORDS) << endl;
  statsRecorder.add("Total Codewords", CodewordT::TOTAL_CODEWORDS);

  cout << "Initial guess: " << CodewordT{packedInitialGuess} << endl;
  statsRecorder.add("Initial Guess", CodewordT{packedInitialGuess});

  auto startTime = chrono::high_resolution_clock::now();

  solver.playAllGames(packedInitialGuess);

  auto endTime = chrono::high_resolution_clock::now();
  double averageTurns = (double)solver.getTotalTurns() / CodewordT::TOTAL_CODEWORDS;
  printf("Average number of turns was %.4f\n", averageTurns);
  statsRecorder.add("Average Turns", averageTurns);
  cout << "Maximum number of turns over all possible secrets was " << solver.getMaxDepth() << endl;
  statsRecorder.add("Max Turns", solver.getMaxDepth());
  //  statsRecorder.add("Max Secret", maxSecret);
  chrono::duration<float, milli> elapsedMS = endTime - startTime;
  auto elapsedS = elapsedMS.count() / 1000;
  cout << "Elapsed time " << commaString(elapsedS) << "s" << endl;
  statsRecorder.add("Elapsed (s)", elapsedS);
  //  statsRecorder.add("Average Game Time (ms)", avgGameTimeMS);

  // mmmfixme: need these two back!
  //  strategy->printStats(elapsedMS);
  //  strategy->recordStats(statsRecorder, elapsedMS);

  printf("\n");
  //  for (int i = 0; i < histogramSize; i++) {
  //    if (histogram[i] > 0) {
  //      printf("%2d: %s ", i, commaString(histogram[i]).c_str());
  //    }
  //    statsRecorder.add(histogramHeaders[i], histogram[i]);
  //  }
  //  printf("\n");

  if (shouldWriteStratFiles) {
    // mmmfixme: need this to dump the strat graph
    //    strategy->dump();
  }

  cout << "Done" << endl;
}

// Find a unique set of initial guesses. Using different digits for the same pattern isn't useful, nor are shuffled
// patterns. For 4p6c the unique initial guesses ae 1111, 1112, 1122, 1123, 1234. Repetitions of the same pattern, such
// as 2222, 2111, 3456, 1223, etc. aren't useful as they yield the same information.
//
// This is pretty brute-force. I feel like I'm missing a clever algorithm or representation for these.
template <uint8_t p, uint8_t c>
set<uint32_t> buildInitialGuessSet() {
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
    for (auto count : colorCounts) {
      for (int i = 0; i < count; i++) {
        uniquePacked = (uniquePacked << 4u) | nextDigit;
      }
      nextDigit++;
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

struct PlayAllGames {
  StatsRecorder& statsRecorder;

  explicit PlayAllGames(StatsRecorder& sr) : statsRecorder(sr) {}

  template <typename Solver>
  void runSolver() {
    using SolverConfig = typename Solver::SolverConfig;
    runSingleSolver<Solver>(
        statsRecorder,
        SolverConfig::ALGO::template presetInitialGuess<SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT>());
  }
};

struct PlayAllGamesWithAllInitialGuesses {
  StatsRecorder& statsRecorder;

  explicit PlayAllGamesWithAllInitialGuesses(StatsRecorder& sr) : statsRecorder(sr) {}

  template <typename Solver>
  void runSolver() {
    using SolverConfig = typename Solver::SolverConfig;
    set<uint32_t> igs = buildInitialGuessSet<SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT>();

    for (auto ig : igs) {
      runSingleSolver<Solver>(statsRecorder, ig);
    }
  }
};

template <bool shouldRun>
void playSingleGame(StatsRecorder& statsRecorder) {
  if constexpr (shouldRun) {
    using gameSolver =
        SingleGameSolver<SolverConfig<singleGamePinCount, singleGameColorCount, singleGameLog, SingleGameAlgo>>;
    PlayAllGames{statsRecorder}.template runSolver<gameSolver>();
  }
}

template <bool shouldRun>
void playMultipleGames(StatsRecorder& statsRecorder) {
  if constexpr (shouldRun) {
    using namespace ss;
    using gameConfigs = solver_config_list<MultiGamePins, MultiGameColors, MultiGameAlgos, multiGameLog>;
    using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
    run_multiple_solvers(gameSolvers::type{}, PlayAllGames(statsRecorder));
  }
}

template <bool shouldRun>
void playMultipleGamesWithInitialGuesses(StatsRecorder& statsRecorder) {
  if constexpr (shouldRun) {
    using namespace ss;
    using gameConfigs = solver_config_list<MultiGamePins, MultiGameColors, MultiGameAlgos, multiGameLog>;
    using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
    run_multiple_solvers(gameSolvers::type{}, PlayAllGamesWithAllInitialGuesses(statsRecorder));
  }
}

int main(int argc, const char* argv[]) {
  runUnitTests<shouldRunTests>();

  setupHistogramHeaders();

  StatsRecorder statsRecorder;
  statsRecorder.addAll("Git Branch", MASTERMIND_GIT_BRANCH);
  statsRecorder.addAll("Git Commit Hash", MASTERMIND_GIT_COMMIT_HASH);
  statsRecorder.addAll("Git Commit Date", MASTERMIND_GIT_COMMIT_DATE);
  string csvTag;  // mmmfixme

  playSingleGame<shouldPlaySingleGame>(statsRecorder);

  playMultipleGames<shouldPlayMultipleGames>(statsRecorder);

  playMultipleGamesWithInitialGuesses<shouldFindBestFirstGuesses>(statsRecorder);

  tm t = {};
  istringstream ss(MASTERMIND_GIT_COMMIT_DATE);
  ss >> get_time(&t, "%Y-%m-%d %H:%M:%S");
  stringstream fs;
  fs << "mastermind_run_stats_" << csvTag << put_time(&t, "%Y%m%d_%H%M%S") << "_" << MASTERMIND_GIT_COMMIT_HASH
     << ".csv";
  statsRecorder.writeStats(fs.str());

  return 0;
}
