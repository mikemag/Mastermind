// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <chrono>
#include <filesystem>
#include <set>
#include <unordered_set>

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
using MultiGameAlgos = ss::algo_list<Algos::Knuth>;
using MultiGamePins = ss::pin_counts<4>;
using MultiGameColors = ss::color_counts<6>;
// using MultiGameAlgos = ss::algo_list<Algos::Knuth, Algos::MostParts, Algos::ExpectedSize, Algos::Entropy>;
// using MultiGamePins = ss::pin_counts<2, 3, 4>;
// using MultiGameColors = ss::color_counts<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12>;
static constexpr bool multiGameLog = false;
// static constexpr const char* fileTag = "_aa_7p_2-9c_8p_2-7c";
static constexpr const char* fileTag = "";

// Initial guess exploration, plays the same games as the multi game config above
static constexpr bool shouldFindBestFirstGuesses = false;
static constexpr bool shouldFindBestFirstSpecificGuesses = false;

// Misc config
static constexpr bool shouldRunTests = true;  // Run unit tests and play Knuth's game
static constexpr bool shouldWriteStratFiles = false;
static constexpr bool shouldSkipCompletedGames = false;  // Load stats in current dir and skip completed games

static constexpr auto statsFilenamePrefix = "mastermind_run_stats_";
static constexpr auto pinCountTag = "Pin Count";
static constexpr auto colorCountTag = "Color Count";
static constexpr auto strategyTag = "Strategy";
static constexpr auto solverTag = "Solver";
static constexpr auto initialGuessTag = "Initial Guess";

// Previous runs visible in the current dir
static std::unordered_set<string> previousRuns;

string buildRunKey(int pc, int cc, const string& strategy, const string& solver, const string& initialGuess) {
  stringstream k;
  k << pc << cc << strategy << solver << initialGuess;
  return k.str();
}

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

template <typename Solver>
void validateSolutions(Solver& solver, json& validSolutions) {
  using SolverConfig = typename Solver::SolverConfig;
  using CodewordT = typename SolverConfig::CodewordT;

  if (validSolutions.contains(SolverConfig::ALGO::name)) {
    auto s = validSolutions[SolverConfig::ALGO::name];
    if (s.contains(to_string(SolverConfig::PIN_COUNT))) {
      auto p = s[to_string(SolverConfig::PIN_COUNT)];
      if (p.contains(to_string(SolverConfig::COLOR_COUNT))) {
        auto vs = p[to_string(SolverConfig::COLOR_COUNT)];
        bool valid = true;

        unsigned long long int totalTurns = vs["total_turns"];
        if (solver.getTotalTurns() != totalTurns) {
          printf("ERROR: Total turns doesn't match, expect %llu (%.4f), actual %llu (%.4f)\n", totalTurns,
                 (double)totalTurns / CodewordT::TOTAL_CODEWORDS, solver.getTotalTurns(),
                 (double)solver.getTotalTurns() / CodewordT::TOTAL_CODEWORDS);
          valid = false;
        }

        uint maxTurns = vs["max_turns"];
        if (solver.getMaxDepth() != maxTurns) {
          printf("ERROR: Max turns doesn't match, expect %u, actual %u\n", maxTurns, solver.getMaxDepth());
          valid = false;
        }

        auto printGuesses = [](const vector<uint32_t>& guesses) {
          printf("%x", guesses[0]);
          for (int i = 1; i < guesses.size(); i++) printf(", %x", guesses[i]);
        };

        vector<vector<string>> sampleGames = vs["sample_games"];
        for (vector<string>& g : sampleGames) {
          vector<uint32_t> gi(g.size());
          std::transform(g.begin(), g.end(), gi.begin(), [&](const string& s) { return stoi(s, 0, 16); });
          auto guesses = solver.getGuessesForGame(gi.back());
          if (guesses != gi) {
            printf("ERROR: Solution for secret %x, %dp%dc game, algo '%s', solver '%s', %ld moves: ", gi.back(),
                   SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name, Solver::name,
                   guesses.size());
            printGuesses(guesses);
            printf(" <-- **WRONG ANSWER**, should be ");
            printGuesses(gi);
            printf("\n");
            valid = false;
          }
        }
        if (valid) {
          printf("Verified solution.\n");
        }
        return;
      }
    }
  }

  printf("Note: no saved solution to verify against.\n");
}

template <typename Solver>
void runSingleSolver(StatsRecorder& statsRecorder, json& validSolutions, uint32_t packedInitialGuess) {
  using SolverConfig = typename Solver::SolverConfig;
  using CodewordT = typename SolverConfig::CodewordT;

  if constexpr (shouldSkipCompletedGames) {
    auto runKey = buildRunKey(SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name,
                              Solver::name, hexString(packedInitialGuess));
    if (auto iter = previousRuns.find(runKey); iter != previousRuns.end()) {
      printf("Skipping completed run for %dp%dc '%s', initial guess '%s', and solver '%s'.\n\n",
             SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name,
             hexString(packedInitialGuess).c_str(), Solver::name);
      return;
    }
  }

  Solver solver;

  statsRecorder.newRun();

  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&now_time), "%FT%T%z");
  statsRecorder.add("Time", ss.str());

  printf("Playing all %d pin %d color games using algorithm '%s' and solver '%s' for every possible secret...\n",
         SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT, SolverConfig::ALGO::name, Solver::name);

  if (solver.usesGPU() && statsRecorder.gpuInfo.hasGPU()) {
    printf("Using GPU %s\n", to_string(statsRecorder.gpuInfo.info["GPU Name"]).c_str());
  }

  statsRecorder.add(pinCountTag, (int)SolverConfig::PIN_COUNT);
  statsRecorder.add(colorCountTag, (int)SolverConfig::COLOR_COUNT);
  statsRecorder.add(strategyTag, SolverConfig::ALGO::name);
  statsRecorder.add(solverTag, Solver::name);

  cout << "Total codewords: " << commaString(CodewordT::TOTAL_CODEWORDS) << endl;
  statsRecorder.add("Total Codewords", CodewordT::TOTAL_CODEWORDS);

  cout << "Initial guess: " << CodewordT{packedInitialGuess} << endl;
  statsRecorder.add(initialGuessTag, hexString(packedInitialGuess));

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

  validateSolutions(solver, validSolutions);

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
  json& validSolutions;

  explicit PlayAllGames(StatsRecorder& sr, json& vs) : statsRecorder(sr), validSolutions(vs) {}

  template <typename Solver>
  void runSolver() {
    using SolverConfig = typename Solver::SolverConfig;
    runSingleSolver<Solver>(
        statsRecorder, validSolutions,
        SolverConfig::ALGO::template presetInitialGuess<SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT>());
  }
};

struct PlayAllGamesWithAllInitialGuesses {
  StatsRecorder& statsRecorder;
  json& validSolutions;

  explicit PlayAllGamesWithAllInitialGuesses(StatsRecorder& sr, json& vs) : statsRecorder(sr), validSolutions(vs) {}

  template <typename Solver>
  void runSolver() {
    using SolverConfig = typename Solver::SolverConfig;
    set<uint32_t> igs = buildInitialGuessSet<SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT>();

    for (auto ig : igs) {
      runSingleSolver<Solver>(statsRecorder, validSolutions, ig);
    }
  }
};

template <bool shouldRun>
void playSingleGame(StatsRecorder& statsRecorder, json& validSolutions) {
  if constexpr (shouldRun) {
    using gameSolver =
        SingleGameSolver<SolverConfig<singleGamePinCount, singleGameColorCount, singleGameLog, SingleGameAlgo>>;
    PlayAllGames{statsRecorder, validSolutions}.template runSolver<gameSolver>();
  }
}

template <bool shouldRun>
void playMultipleGames(StatsRecorder& statsRecorder, json& validSolutions) {
  if constexpr (shouldRun) {
    using namespace ss;
    using gameConfigs = solver_config_list<MultiGamePins, MultiGameColors, MultiGameAlgos, multiGameLog>;
    using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
    run_multiple_solvers(gameSolvers::type{}, PlayAllGames(statsRecorder, validSolutions));
  }
}

template <bool shouldRun>
void playMultipleSpecificGames(StatsRecorder& statsRecorder, json& validSolutions) {
  if constexpr (shouldRun) {
    using namespace ss;
    {
      using gameConfigs =
          solver_config_list<ss::pin_counts<7>, ss::color_counts<2, 3, 4, 5, 6, 7, 8, 9>, MultiGameAlgos, multiGameLog>;
      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
      run_multiple_solvers(gameSolvers::type{}, PlayAllGames(statsRecorder, validSolutions));
    }
    {
      using gameConfigs =
          solver_config_list<ss::pin_counts<8>, ss::color_counts<2, 3, 4, 5, 6, 7>, MultiGameAlgos, multiGameLog>;
      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
      run_multiple_solvers(gameSolvers::type{}, PlayAllGames(statsRecorder, validSolutions));
    }
  }
}

template <bool shouldRun>
void playMultipleGamesWithInitialGuesses(StatsRecorder& statsRecorder, json& validSolutions) {
  if constexpr (shouldRun) {
    using namespace ss;
    using gameConfigs = solver_config_list<MultiGamePins, MultiGameColors, MultiGameAlgos, multiGameLog>;
    using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
    run_multiple_solvers(gameSolvers::type{}, PlayAllGamesWithAllInitialGuesses(statsRecorder, validSolutions));
  }
}

template <bool shouldRun>
void playMultipleSpecificGamesWithInitialGuesses(StatsRecorder& statsRecorder, json& validSolutions) {
  if constexpr (shouldRun) {
    using namespace ss;
    {
      using gameConfigs =
          solver_config_list<ss::pin_counts<8>, ss::color_counts<2, 3, 4, 5, 6>, MultiGameAlgos, multiGameLog>;
      using gameSolvers = build_solvers<MultiGameSolver, gameConfigs::type>;
      run_multiple_solvers(gameSolvers::type{}, PlayAllGamesWithAllInitialGuesses(statsRecorder, validSolutions));
    }
  }
}

void readAllStats() {
  cout << "Reading existing stats files..." << endl;

  for (const auto& entry : std::filesystem::directory_iterator(".")) {
    string filename = entry.path().filename();

    if (filename.rfind(statsFilenamePrefix, 0) == 0) {
      cout << filename << endl;

      ifstream i(filename);
      std::string line;
      while (std::getline(i, line)) {
        if (line == "[" || line == "]") continue;
        if (line[0] == ',') line[0] = ' ';
        try {
          auto j = json::parse(line);

          if (j.contains("run")) {
            auto run = j["run"];
            auto runKey = buildRunKey(run[pinCountTag], run[colorCountTag], run[strategyTag], run[solverTag],
                                      run[initialGuessTag]);
            previousRuns.insert(runKey);
          }

        } catch (json::parse_error& ex) {
          // Just drop bad lines and re-do those runs.
        }
      }
    }
  }

  cout << "Done reading previous runs." << endl << endl;
}

int main(int argc, const char* argv[]) {
  runUnitTests<shouldRunTests>();

  if constexpr (shouldSkipCompletedGames) {
    readAllStats();
  }

  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  stringstream fs;
  fs << statsFilenamePrefix << put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S");
  auto st = std::getenv("MASTERMIND_STATS_TAG");
  if (st) {
    fs << "_" << st;
  }
  fs << fileTag << ".json";
  cout << "Will write stats to: " << fs.str() << endl << endl;
  StatsRecorder statsRecorder(fs.str());

  std::ifstream vsf("valid_solutions.json");
  json validSolutions = json::parse(vsf);

  playSingleGame<shouldPlaySingleGame>(statsRecorder, validSolutions);

  playMultipleGames<shouldPlayMultipleGames>(statsRecorder, validSolutions);
  playMultipleSpecificGames<shouldPlayMultipleSpecificGames>(statsRecorder, validSolutions);

  playMultipleGamesWithInitialGuesses<shouldFindBestFirstGuesses>(statsRecorder, validSolutions);
  playMultipleSpecificGamesWithInitialGuesses<shouldFindBestFirstSpecificGuesses>(statsRecorder, validSolutions);

  cout << "Complete." << endl;
  return 0;
}
