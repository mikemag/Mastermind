// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <chrono>
#include <set>
#include <typeindex>

#include "codeword.hpp"
#include "mastermind_config.h"
#include "score.hpp"
#include "solver.hpp"
#include "solver_cpu_opt.hpp"
#include "solver_cpu_reference.hpp"
#include "solver_sets.hpp"

#ifdef __CUDACC__
#include "solver_cuda.hpp"
#endif

using namespace std;
namespace ss = solver_sets;

// Mastermind
//
// Play the game Mastermind, which is to guess a sequence of pins with different colors (the "secret") with feedback
// about how many you have guessed correctly, and how many are in the right place. This will play the game for every
// possible secret and tell us the average and maximum number of tries needed across all of them.
//
// See the README.md for more details, results, and links to more docs describing what's going on here.

#ifdef __CUDACC__
template <typename T>
using DefaultSolver = SolverCUDA<T>;
#else
template <typename T>
using DefaultSolver = SolverCPUFaster<T>;
#endif

// Config for a single game
static constexpr bool shouldPlaySingleGame = true;
template <typename T>
using SingleGameSolver = DefaultSolver<T>;
using SingleGameAlgo = Algos::Knuth;
static constexpr uint8_t singleGamePinCount = 4;    // 1-8, 4 is classic
static constexpr uint8_t singleGameColorCount = 6;  // 1-15, 6 is classic
static constexpr bool singleGameLog = true;

// Config for playing a set of games
static constexpr bool shouldPlayMultipleGames = false;
static constexpr bool shouldPlayMultipleSpecificGames = false;
template <typename T>
using MultiGameSolver = DefaultSolver<T>;
using MultiGameAlgos = ss::algo_list<Algos::Knuth, Algos::MostParts, Algos::ExpectedSize, Algos::Entropy>;
using MultiGamePins = ss::pin_counts<6>;
using MultiGameColors = ss::color_counts<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12>;
static constexpr bool multiGameLog = false;
// static constexpr const char* fileTag = "_aa_7p_2-9c_8p_2-7c";
static constexpr const char* fileTag = "";

// Initial guess exploration, plays the same games as the multi game config above
static constexpr bool shouldFindBestFirstGuesses = false;
static constexpr bool shouldFindBestFirstSpecificGuesses = false;

// Misc config
static constexpr bool shouldRunTests = true;  // Run unit tests and play Knuth's game
static constexpr bool shouldWriteStratFiles = false;

template <bool shouldRun>
void runUnitTests() {
  if constexpr (shouldRun) {
    // Test cases from Miyoshi
    constexpr Codeword<4, 10> testSecret(0x6684);
    bool success = true;
    success &= (testSecret.score(Codeword<4, 10>(0x5555)) == Score(0, 0));
    success &= (testSecret.score(Codeword<4, 10>(0x6666)) == Score(2, 0));
    success &= (testSecret.score(Codeword<4, 10>(0x5123)) == Score(0, 0));
    success &= (testSecret.score(Codeword<4, 10>(0x4567)) == Score(0, 2));
    success &= (testSecret.score(Codeword<4, 10>(0x4589)) == Score(1, 1));
    success &= (testSecret.score(Codeword<4, 10>(0x6755)) == Score(1, 0));
    success &= (testSecret.score(Codeword<4, 10>(0x5798)) == Score(0, 1));
    success &= (testSecret.score(Codeword<4, 10>(0x6484)) == Score(3, 0));
    success &= (testSecret.score(Codeword<4, 10>(0x6485)) == Score(2, 1));
    success &= (testSecret.score(Codeword<4, 10>(0x6884)) == Score(3, 0));
    success &= (testSecret.score(Codeword<4, 10>(0x6684)) == Score(4, 0));

    // Three extra tests to detect subtly broken scoring functions.
    success &= (testSecret.score(Codeword<4, 10>(0x8468)) == Score(0, 3));
    success &= (testSecret.score(Codeword<4, 10>(0x8866)) == Score(0, 3));
    success &= (testSecret.score(Codeword<4, 10>(0x8466)) == Score(0, 4));

    if (success) {
      printf("Tests pass\n\n");
    } else {
      printf("Some tests failed!\n");
      exit(-1);
    }
  }
}

// A little struct to hold info about a valid result from a solver.
struct ValidSolution {
  uint maxTurns;
  unsigned long long int totalTurns;

  struct SingleGame {
    uint32_t secret;
    vector<uint32_t> correctGuesses;
  };

  vector<SingleGame> games;
};

// List of known solutions to verify our runs
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT, typename ALGO>
struct ValidSolutionsKey {};

static std::map<std::type_index, ValidSolution> validSolutions = {
    {typeid(ValidSolutionsKey<4, 6, Algos::Knuth>),
     {5,
      5801,
      {
          {0x3632, {0x1122, 0x1344, 0x3526, 0x1462, 0x3632}},
          {0x1111, {0x1122, 0x1234, 0x1315, 0x1111}},
      }}},
    {typeid(ValidSolutionsKey<4, 6, Algos::MostParts>),
     {6,
      5668,
      {
          {0x3632, {0x1123, 0x2344, 0x3255, 0x3632}},
          {0x1111, {0x1123, 0x1425, 0x2326, 0x1111}},
      }}},
    {typeid(ValidSolutionsKey<5, 8, Algos::Knuth>),
     {7,
      183775,
      {
          {0x34567, {0x11223, 0x34455, 0x53657, 0x35856, 0x34567}},
      }}},
    {typeid(ValidSolutionsKey<7, 7, Algos::Knuth>),
     {8,
      5124234,
      {
          {0x4422334, {0x1122334, 0x1225445, 0x1512361, 0x4322334, 0x4422334}},
      }}},
    {typeid(ValidSolutionsKey<7, 7, Algos::MostParts>),
     {10,
      5073674,
      {
          {0x4422334, {0x1112233, 0x1121444, 0x1215156, 0x2342434, 0x2432443, 0x4422334}},
      }}},
    {typeid(ValidSolutionsKey<8, 5, Algos::Knuth>),
     {8,
      2281524,
      {
          {0x11223344, {0x11122334, 0x14412131, 0x11223521, 0x11223344}},
      }}},
};

template <typename Solver>
void validateSolutions(Solver& solver) {
  using SolverConfig = typename Solver::SolverConfig;
  using CodewordT = typename SolverConfig::CodewordT;
  using vsKey = ValidSolutionsKey<SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, typename SolverConfig::ALGO>;
  auto vsIter = validSolutions.find(typeid(vsKey));
  if (vsIter != validSolutions.end()) {
    const ValidSolution& vs = vsIter->second;

    if (solver.getTotalTurns() != vs.totalTurns) {
      printf("ERROR: Total turns doesn't match, expect %llu (%.4f), actual %llu (%.4f)\n", vs.totalTurns,
             (double)vs.totalTurns / CodewordT::TOTAL_CODEWORDS, solver.getTotalTurns(),
             (double)solver.getTotalTurns() / CodewordT::TOTAL_CODEWORDS);
    }

    if (solver.getMaxDepth() != vs.maxTurns) {
      printf("ERROR: Max turns doesn't match, expect %u, actual %u\n", vs.maxTurns, solver.getMaxDepth());
    }

    auto printGuesses = [](const vector<uint32_t>& guesses) {
      printf("%x", guesses[0]);
      for (int i = 1; i < guesses.size(); i++) printf(", %x", guesses[i]);
    };

    for (auto& g : vs.games) {
      auto guesses = solver.getGuessesForGame(g.secret);
      if (guesses != g.correctGuesses) {
        printf("ERROR: Solution for secret %x, %dp%dc game, algo '%s', solver '%s', %ld moves: ", g.secret,
               SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name, Solver::name,
               guesses.size());
        printGuesses(guesses);
        printf(" <-- **WRONG ANSWER**, should be ");
        printGuesses(g.correctGuesses);
        printf("\n");
      }
    }
  }
}

template <typename Solver>
void runSingleSolver(StatsRecorder& statsRecorder, uint32_t packedInitialGuess) {
  using SolverConfig = typename Solver::SolverConfig;
  using CodewordT = typename SolverConfig::CodewordT;

  Solver solver;

  statsRecorder.newRun();
  printf("Playing all %d pin %d color games using algorithm '%s' and solver '%s' for every possible secret...\n",
         SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name, Solver::name);

  if (solver.usesGPU() && statsRecorder.gpuInfo.hasGPU()) {
    printf("Using GPU %s\n", to_string(statsRecorder.gpuInfo.info["GPU Name"]).c_str());
  }

  statsRecorder.add("Pin Count", (int)SolverConfig::PIN_COUNT);
  statsRecorder.add("Color Count", (int)SolverConfig::COLOR_COUNT);
  statsRecorder.add("Strategy", SolverConfig::ALGO::name);
  statsRecorder.add("Solver", Solver::name);

  cout << "Total codewords: " << commaString(CodewordT::TOTAL_CODEWORDS) << endl;
  statsRecorder.add("Total Codewords", CodewordT::TOTAL_CODEWORDS);

  cout << "Initial guess: " << CodewordT{packedInitialGuess} << endl;
  statsRecorder.add("Initial Guess", hexString(packedInitialGuess));

  auto elapsed = solver.playAllGames(packedInitialGuess);

  double averageTurns = (double)solver.getTotalTurns() / CodewordT::TOTAL_CODEWORDS;
  printf("Average number of turns was %.4f\n", averageTurns);
  statsRecorder.add("Average Turns", averageTurns);
  cout << "Maximum number of turns over all possible secrets was " << solver.getMaxDepth() << endl;
  statsRecorder.add("Max Turns", solver.getMaxDepth());
  statsRecorder.add("Total Turns", solver.getTotalTurns());
  chrono::duration<float> elapsedS = elapsed;
  cout << "Elapsed time " << commaString(elapsedS.count()) << "s" << endl;
  statsRecorder.add("Elapsed (s)", elapsedS.count());

  // Grab a sample game to record and use as a test case. Random pick: whatever game is 42% into the list of all games.
  auto& allCodewords = SolverConfig::CodewordT::getAllCodewords();
  auto guesses = solver.getGuessesForGame(allCodewords[allCodewords.size() * 0.42].packedCodeword());
  vector<string> sampleMoves;
  std::transform(guesses.begin(), guesses.end(), std::back_inserter(sampleMoves), hexString);
  statsRecorder.add("Sample Game", json(sampleMoves));

  cout << endl;
  solver.printStats();
  solver.recordStats(statsRecorder);

  validateSolutions(solver);

  if (shouldWriteStratFiles) {
    solver.dump();
  }

  cout << "Done" << endl << endl;
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
void playMultipleSpecificGames(StatsRecorder& statsRecorder) {
  if constexpr (shouldRun) {
    using namespace ss;
    {
      using gameConfigs =
          solver_config_list<ss::pin_counts<7>, ss::color_counts<2, 3, 4, 5, 6, 7, 8, 9>, MultiGameAlgos, multiGameLog>;
      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
      run_multiple_solvers(gameSolvers::type{}, PlayAllGames(statsRecorder));
    }
    {
      using gameConfigs =
          solver_config_list<ss::pin_counts<8>, ss::color_counts<2, 3, 4, 5, 6, 7>, MultiGameAlgos, multiGameLog>;
      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
      run_multiple_solvers(gameSolvers::type{}, PlayAllGames(statsRecorder));
    }
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

template <bool shouldRun>
void playMultipleSpecificGamesWithInitialGuesses(StatsRecorder& statsRecorder) {
  if constexpr (shouldRun) {
    using namespace ss;
    //    {
    //      using gameConfigs = solver_config_list<ss::pin_counts<6>, ss::color_counts<2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    //      12>,
    //                                             MultiGameAlgos, multiGameLog>;
    //      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
    //      run_multiple_solvers(gameSolvers::type{}, PlayAllGamesWithAllInitialGuesses(statsRecorder));
    //    }
    //    {
    //      using gameConfigs =
    //          solver_config_list<ss::pin_counts<7>, ss::color_counts<2, 3, 4, 5, 6, 7, 8, 9>, MultiGameAlgos,
    //          multiGameLog>;
    //      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
    //      run_multiple_solvers(gameSolvers::type{}, PlayAllGamesWithAllInitialGuesses(statsRecorder));
    //    }
    {
      using gameConfigs =
          solver_config_list<ss::pin_counts<8>, ss::color_counts<2, 3, 4, 5, 6>, MultiGameAlgos, multiGameLog>;
      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
      run_multiple_solvers(gameSolvers::type{}, PlayAllGamesWithAllInitialGuesses(statsRecorder));
    }
  }
}

int main(int argc, const char* argv[]) {
  runUnitTests<shouldRunTests>();

  tm t = {};
  istringstream ss(MASTERMIND_GIT_COMMIT_DATE);
  ss >> get_time(&t, "%Y-%m-%d %H:%M:%S");
  stringstream fs;
  fs << "mastermind_run_stats_" << put_time(&t, "%Y%m%d_%H%M%S") << "_" << MASTERMIND_GIT_COMMIT_HASH << fileTag
     << ".json";
  StatsRecorder statsRecorder(fs.str());
  statsRecorder.addAll("Git Branch", MASTERMIND_GIT_BRANCH);
  statsRecorder.addAll("Git Commit Hash", MASTERMIND_GIT_COMMIT_HASH);
  statsRecorder.addAll("Git Commit Date", MASTERMIND_GIT_COMMIT_DATE);

  playSingleGame<shouldPlaySingleGame>(statsRecorder);

  playMultipleGames<shouldPlayMultipleGames>(statsRecorder);
  playMultipleSpecificGames<shouldPlayMultipleSpecificGames>(statsRecorder);

  playMultipleGamesWithInitialGuesses<shouldFindBestFirstGuesses>(statsRecorder);
  playMultipleSpecificGamesWithInitialGuesses<shouldFindBestFirstSpecificGuesses>(statsRecorder);

  cout << "Complete." << endl;
  return 0;
}
