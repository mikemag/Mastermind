// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <unordered_map>
#include <vector>

#include "codeword.hpp"
#include "score.hpp"
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

template <uint8_t p, uint8_t c, bool l>
class Strategy {
 public:
  Strategy() : savedPossibleSolutions(Codeword<p, c>::getAllCodewords()) { rootData = make_shared<StrategyRootData>(); }

  explicit Strategy(Codeword<p, c> guess) : savedPossibleSolutions(Codeword<p, c>::getAllCodewords()) {
    this->guess = guess;
    rootData = make_shared<StrategyRootData>();
  }

  virtual std::string getName() const = 0;
  Codeword<p, c> currentGuess() const { return guess; }

  uint32_t findSecret(Codeword<p, c> secret, int depth = 0);

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
  Codeword<p, c> guess;
  std::vector<Codeword<p, c>> possibleSolutions;

  shared_ptr<StrategyRootData> rootData = nullptr;

  Strategy(Strategy<p, c, l> &parent, Codeword<p, c> nextGuess, std::vector<Codeword<p, c>> &nextPossibleSolutions)
      : savedPossibleSolutions(std::move(nextPossibleSolutions)), guess(nextGuess), rootData(parent.rootData) {}

  void removeImpossibleSolutions(Score r);
  virtual Codeword<p, c> selectNextGuess() = 0;
  virtual std::shared_ptr<Strategy<p, c, l>> createNewMove(Score r, Codeword<p, c> nextGuess) = 0;

  // Knuth's algorithm uses 1122 to start. Generalize that to half 1's and half 2's for any number of pins.
  constexpr static uint32_t genericInitialGuess = (Codeword<p, c>::onePins >> p / 2 * 4) + Codeword<p, c>::onePins;

  constexpr static uint8_t packedPinsAndColors = (p << 4u) | c;
  constexpr static uint32_t maxScoreSlots = (p << 4u) + 1;
  constexpr static int totalScores = (p * (p + 3)) / 2;

  // Optimization control -- for experimentation
  constexpr static bool enableTwoPSShortcut = true;
  constexpr static bool enableSmallPSShortcut = true;
  constexpr static bool enableSmallPSShortcutGPU = true;

 private:
  // The strategy is made up of the next guess to play, and a map of where to go based on the result of that play.
  std::unordered_map<Score, std::shared_ptr<Strategy<p, c, l>>> nextMoves;
  const std::vector<Codeword<p, c>> savedPossibleSolutions;

  Codeword<p, c> shortcutSmallSets();

  void dumpRoot(std::ofstream &graphStream);
  void dump(std::ofstream &graphStream);
  void dumpChildren(std::ofstream &graphStream);
};

// Data shared by all Strategy nodes within a strategy tree
struct StrategyRootData {
  uint64_t scoreCounterCPU = 0;
  uint64_t scoreCounterGPU = 0;

  // Optimization metrics -- measuring experimentation
  constexpr static bool enableTwoPSMetrics = true;
  ExperimentCounter<enableTwoPSMetrics> twoPSShortcuts =
      ExperimentCounter<enableTwoPSMetrics>("Exp: Size 2 PS Shortcuts");

  constexpr static bool enableSmallPSMetrics = true;
  ExperimentCounter<enableSmallPSMetrics> smallPSHighShortcuts =
      ExperimentCounter<enableSmallPSMetrics>("Exp: Small PS High Shortcuts");
  ExperimentCounter<enableSmallPSMetrics> smallPSHighWasted =
      ExperimentCounter<enableSmallPSMetrics>("Exp: Small PS High Wasted");
  ExperimentCounter<enableSmallPSMetrics> smallPSHighScores =
      ExperimentCounter<enableSmallPSMetrics>("Exp: Small PS High Scores");
  ExperimentCounter<enableSmallPSMetrics> smallPSInnerShortcuts =
      ExperimentCounter<enableSmallPSMetrics>("Exp: Small PS Inner Shortcuts");
  ExperimentCounter<enableSmallPSMetrics> smallPSInnerWasted =
      ExperimentCounter<enableSmallPSMetrics>("Exp: Small PS Inner Wasted");
  ExperimentCounter<enableSmallPSMetrics> smallPSInnerScoresSkipped =
      ExperimentCounter<enableSmallPSMetrics>("Small PS Inner Scores Skipped");
};

#include "strategy.cpp"
