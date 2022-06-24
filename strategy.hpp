// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

#include "codeword.hpp"
#include "score.hpp"
#include "strategy_config.hpp"
#include "utils.hpp"

// Gameplay Strategy
//
// Methods of playing the game (i.e., selecting guesses), and recording the gameplay strategy for later use and display.
//
// Subclasses of this implement various gameplay strategies, like "first one", "random", or "Knuth".
//
// This is also used to build a tree of plays to make based on previous plays and results. All games start with the same
// guess, which makes the root of the tree. The score received is used to find what to play next via the nextMoves map.
// If there is no entry in the map, then the strategy will do whatever work is necessary (possibly large) to find the
// next play, then add it to the tree. As games are played, the tree gets filled in and playtime decreases.
struct StrategyRootData;

template <typename StrategyConfig>
class Strategy {
 public:
  using CodewordT = Codeword<StrategyConfig::PIN_COUNT, StrategyConfig::COLOR_COUNT>;

  explicit Strategy(CodewordT guess) : savedPossibleSolutions(CodewordT::getAllCodewords()) {
    this->guess = guess;
    rootData = make_shared<StrategyRootData>();
  }

  virtual std::string getName() const = 0;
  CodewordT currentGuess() const { return guess; }

  uint32_t findSecret(CodewordT secret, int depth = 0);

  virtual void printStats(std::chrono::duration<float, std::milli> elapsedMS);
  virtual void recordStats(StatsRecorder &sr, std::chrono::duration<float, std::milli> elapsedMS);

  // Output the strategy for visualization with GraphViz. Copy-and-paste the output file to sites
  // like https://dreampuf.github.io/GraphvizOnline or http://www.webgraphviz.com/. Or install
  // GraphViz locally and run with the following command:
  //
  //   twopi -Tjpg mastermind_strategy_4p6c.gv > mastermind_strategy_4p6c.jpg
  //
  // Parameters for the graph are currently set to convey the point while being reasonably readable
  // in a large JPG.
  void dump();

 protected:
  // These extra members are to allow us to build the strategy lazily, as we play games using any algorithm. nb: these
  // are copies.
  CodewordT guess;
  std::vector<CodewordT> possibleSolutions;

  shared_ptr<StrategyRootData> rootData = nullptr;

  Strategy(Strategy<StrategyConfig> &parent, CodewordT nextGuess, std::vector<CodewordT> &nextPossibleSolutions)
      : savedPossibleSolutions(std::move(nextPossibleSolutions)), guess(nextGuess), rootData(parent.rootData) {}

  void removeImpossibleSolutions(Score r);
  virtual CodewordT selectNextGuess() = 0;
  virtual std::shared_ptr<Strategy<StrategyConfig>> createNewMove(Score r, CodewordT nextGuess) = 0;

  constexpr static uint32_t MAX_SCORE_SLOTS = (StrategyConfig::PIN_COUNT << 4u) + 1;

  // Optimization control -- for experimentation
  constexpr static bool ENABLE_TWO_PS_SHORTCUT = true;
  constexpr static bool ENABLE_SMALL_PS_SHORTCUT = true;
  constexpr static bool ENABLE_SMALL_PS_SHORTCUT_GPU = true;

 private:
  // The strategy is made up of the next guess to play, and a map of where to go based on the result of that play.
  std::unordered_map<Score, std::shared_ptr<Strategy<StrategyConfig>>> nextMoves;
  const std::vector<CodewordT> savedPossibleSolutions;

  CodewordT shortcutSmallSets();

  void dumpRoot(std::ofstream &graphStream);
  void dump(std::ofstream &graphStream);
  void dumpChildren(std::ofstream &graphStream);
};

// Data shared by all Strategy nodes within a strategy tree
struct StrategyRootData {
  uint64_t scoreCounterCPU = 0;
  uint64_t scoreCounterGPU = 0;

  // Optimization metrics -- measuring experimentation
  constexpr static bool ENABLE_TWO_PS_METRICS = false;
  ExperimentCounter<ENABLE_TWO_PS_METRICS> twoPSShortcuts =
      ExperimentCounter<ENABLE_TWO_PS_METRICS>("Exp: Size 2 PS Shortcuts");

  constexpr static bool ENABLE_SMALL_PS_METRICS = false;
  ExperimentCounter<ENABLE_SMALL_PS_METRICS> smallPSHighShortcuts =
      ExperimentCounter<ENABLE_SMALL_PS_METRICS>("Exp: Small PS High Shortcuts");
  ExperimentCounter<ENABLE_SMALL_PS_METRICS> smallPSHighWasted =
      ExperimentCounter<ENABLE_SMALL_PS_METRICS>("Exp: Small PS High Wasted");
  ExperimentCounter<ENABLE_SMALL_PS_METRICS> smallPSHighScores =
      ExperimentCounter<ENABLE_SMALL_PS_METRICS>("Exp: Small PS High Scores");
  ExperimentCounter<ENABLE_SMALL_PS_METRICS> smallPSInnerShortcuts =
      ExperimentCounter<ENABLE_SMALL_PS_METRICS>("Exp: Small PS Inner Shortcuts");
  ExperimentCounter<ENABLE_SMALL_PS_METRICS> smallPSInnerWasted =
      ExperimentCounter<ENABLE_SMALL_PS_METRICS>("Exp: Small PS Inner Wasted");
  ExperimentCounter<ENABLE_SMALL_PS_METRICS> smallPSInnerScoresSkipped =
      ExperimentCounter<ENABLE_SMALL_PS_METRICS>("Exp: Small PS Inner Scores Skipped");
};

#include "strategy.cpp"
