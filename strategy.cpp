// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>

#include "utils.hpp"

using namespace std;

// The core of the gameplay, this does one level of the search for a secret based on the algorithm implemented in
// various subclasses. Returns the number of moves needed to find the secret.
template <typename StrategyConfig>
uint32_t Strategy<StrategyConfig>::findSecret(const CodewordT &secret, int depth) {
  if (depth == 0) {
    if (StrategyConfig::LOG) {
      cout << "Starting search for secret " << secret << ", initial guess is " << guess << " with "
           << commaString(savedPossibleSolutions.size()) << " possibilities." << endl;
    }
  }

  // Play a guess and see how we did.
  Score r = secret.score(guess);
  rootData->scoreCounterCPU++;
  depth += 1;
  if (StrategyConfig::LOG) {
    cout << endl << "Tried guess " << guess << " against secret " << secret << " => " << r << endl;
  }
  if (r == CodewordT::WINNING_SCORE) {
    if (StrategyConfig::LOG) {
      cout << "Solution found after " << depth << " moves." << endl;
    }
    return depth;
  }

  // Do we already have a move based on the result?
  shared_ptr<Strategy<StrategyConfig>> next = nextMoves[r];
  if (next) {
    if (StrategyConfig::LOG) {
      cout << "Following saved strategy for next move..." << endl;
    }
    return next->findSecret(secret, depth);
  }

  // Compute a new move given the result of our previous guess
  possibleSolutions = savedPossibleSolutions;
  removeImpossibleSolutions(r);

  // Figure out what our next guess should be
  CodewordT nextGuess;
  if (possibleSolutions.size() == 1) {
    nextGuess = possibleSolutions.front();
    possibleSolutions.clear();
    if (StrategyConfig::LOG) {
      cout << "Only remaining solution must be correct: " << nextGuess << endl;
    }
  } else if (ENABLE_TWO_PS_SHORTCUT && possibleSolutions.size() == 2) {
    nextGuess = possibleSolutions.front();
    possibleSolutions.erase(begin(possibleSolutions));
    if (StrategyConfig::LOG) {
      cout << "Only two solutions remain, selecting the first one blindly: " << nextGuess << endl;
    }
    ++rootData->twoPSShortcuts;
  } else if (ENABLE_SMALL_PS_SHORTCUT && possibleSolutions.size() <= StrategyConfig::TOTAL_SCORES) {
    nextGuess = shortcutSmallSets();
  } else {
    nextGuess = selectNextGuess();
  }

  // Add the new move to this strategy and follow it
  next = createNewMove(r, nextGuess);
  nextMoves[r] = next;
  return next->findSecret(secret, depth);
}

// "5. Otherwise, remove from S any code that would not give the same response if it (the guess) were the code
// (secret)." -- from the description of Knuth's algorithm at https://en.wikipedia.org/wiki/Mastermind_(board_game)
//
// This describes something common to all good solutions: since the scoring function is commutative, and since we know
// the secret remains in our set of possible solutions, we can quickly eliminate lots and lots of solutions on every
// iteration.
template <typename StrategyConfig>
void Strategy<StrategyConfig>::removeImpossibleSolutions(Score r) {
  if (StrategyConfig::LOG) {
    cout << "Removing inconsistent possibilities... ";
  }
  rootData->scoreCounterCPU += possibleSolutions.size();
  possibleSolutions.erase(remove_if(possibleSolutions.begin(), possibleSolutions.end(),
                                    [&](const CodewordT &CodewordT) { return CodewordT.score(guess) != r; }),
                          possibleSolutions.end());

  // This will result in an extra allocation and copy, but it is worth it to keep memory use in check. It actually makes
  // many larger games faster, and is required to be able to play things like 8p7c without running out of memory on a
  // 16GB system.
  possibleSolutions.shrink_to_fit();

  if (StrategyConfig::LOG) {
    cout << possibleSolutions.size() << " remain." << endl;
  }
  if (possibleSolutions.empty()) {
    // This is only possible if there is a bug in our scoring function.
    assert(!"Failed to find a solution.");
    cout << "Failed to find solution." << endl;
    exit(-1);
  }
}

// Optimization from [2]: if the possible solution set is smaller than the number of possible scores, and if one
// codeword can fully discriminate all of the possible solutions (i.e., it produces a different score for each one),
// then play it right away since it will tell us the winner.
//
// This is an interesting shortcut. It doesn't change the results of any of the subsetting algorithms at all: average
// turns, max turns, max secret, and the full histograms all remain precisely the same. What does change is the number
// of scores computed, and the runtime. For 5p8c games, CPU-only, we see these deltas:
//
// Algorithm        Elapsed S            Codewords Scored
// -------------    -----------------    ---------------------
// Knuth            -1.75849 (-20.4%)    -533,638,403 (-13.9%)
// Most Parts       -1.54395 (-18.1%)    -472,952,414 (-12.4%)
// Entropy          -8.1095  (-42.5%)    -496,779,481 (-13.3%)
// Expected Size    -5.57572 (-39.0%)    -528,089,930 (-14.1%)
//
// That's a big difference. This skips a great deal of work in the inner loop for CPU scoring, where it is almost free
// to compute. In the GPU version (not shown), it also skips some work in the inner loop of consuming GPU results, and
// is very cheap to add to the overall GPU time. The shortcut applied in the outer loop saves both CPU and GPU schemes
// quite a lot of work as well with a reasonable ratio of wasted work to work saved.
//
// All-in-all, this is a very nice shortcut for all of these algorithms.

template <typename StrategyConfig>
typename Strategy<StrategyConfig>::CodewordT Strategy<StrategyConfig>::shortcutSmallSets() {
  int subsetSizes[MAX_SCORE_SLOTS];
  for (const auto &psa : possibleSolutions) {
    fill(begin(subsetSizes), end(subsetSizes), 0);
    for (const auto &psb : possibleSolutions) {
      Score r = psa.score(psb);
      subsetSizes[r.result] = 1;
    }
    rootData->smallPSHighScores += possibleSolutions.size();
    rootData->scoreCounterCPU += possibleSolutions.size();
    int totalSubsets = 0;
    for (auto s : subsetSizes) {
      if (s > 0) {
        totalSubsets++;
      }
    }
    if (totalSubsets == possibleSolutions.size()) {
      if (StrategyConfig::LOG) {
        cout << "Selecting fully discriminating guess from PS: " << psa << ", subsets: " << totalSubsets << endl;
      }
      ++rootData->smallPSHighShortcuts;
      return psa;
    }
  }

  // Didn't find a good candidate, fallback
  ++rootData->smallPSHighWasted;
  return selectNextGuess();
}

template <typename StrategyConfig>
void Strategy<StrategyConfig>::printStats(std::chrono::duration<float, std::milli> elapsedMS) {
  cout << "Codeword comparisons: CPU = " << commaString(rootData->scoreCounterCPU)
       << ", GPU = " << commaString(rootData->scoreCounterGPU)
       << ", total = " << commaString(rootData->scoreCounterCPU + rootData->scoreCounterGPU) << endl;

  if (rootData->ENABLE_TWO_PS_METRICS) {
    cout << rootData->twoPSShortcuts << endl;
  }

  if (rootData->ENABLE_SMALL_PS_METRICS) {
    cout << rootData->smallPSHighShortcuts << endl;
    cout << rootData->smallPSHighWasted << endl;
    cout << rootData->smallPSHighScores << endl;
    cout << rootData->smallPSInnerShortcuts << endl;
    cout << rootData->smallPSInnerWasted << endl;
    cout << rootData->smallPSInnerScoresSkipped << endl;
  }
}

template <typename StrategyConfig>
void Strategy<StrategyConfig>::recordStats(StatsRecorder &sr, std::chrono::duration<float, std::milli> elapsedMS) {
  sr.add("CPU Scores", rootData->scoreCounterCPU);
  sr.add("GPU Scores", rootData->scoreCounterGPU);

  if (rootData->ENABLE_TWO_PS_METRICS) {
    rootData->twoPSShortcuts.record(sr);
  }

  if (rootData->ENABLE_SMALL_PS_METRICS) {
    rootData->smallPSHighShortcuts.record(sr);
    rootData->smallPSHighWasted.record(sr);
    rootData->smallPSHighScores.record(sr);
    rootData->smallPSInnerShortcuts.record(sr);
    rootData->smallPSInnerWasted.record(sr);
    rootData->smallPSInnerScoresSkipped.record(sr);
  }
}

// See header for notes on how to use this output. Parameters for the graph are currently set to convey the point while
// being reasonably readable in a large JPG.
template <typename StrategyConfig>
void Strategy<StrategyConfig>::dump() {
  ostringstream fnStream;
  string algName = getName();
  replace(algName.begin(), algName.end(), ' ', '_');
  std::transform(algName.begin(), algName.end(), algName.begin(), ::tolower);
  fnStream << "mastermind_strategy_" << algName << "_" << (int)StrategyConfig::PIN_COUNT << "p"
           << (int)StrategyConfig::COLOR_COUNT << "c.gv";
  string filename = fnStream.str();

  cout << "\nWriting strategy to " << filename << endl;
  ofstream graphStream(filename);
  graphStream << "digraph Mastermind_Strategy_" << getName() << "_" << (int)StrategyConfig::PIN_COUNT << "p"
              << (int)StrategyConfig::COLOR_COUNT << "c";
  graphStream << " {" << endl;
  graphStream << "size=\"40,40\"" << endl;  // Good size for jpgs
  graphStream << "overlap=true" << endl;    // scale is cool, but the result is unreadable
  graphStream << "ranksep=5" << endl;
  graphStream << "ordering=out" << endl;
  graphStream << "node [shape=plaintext]" << endl;
  dumpRoot(graphStream);
  graphStream << "}" << endl;
  graphStream.close();
}

template <typename StrategyConfig>
void Strategy<StrategyConfig>::dumpRoot(ofstream &graphStream) {
  graphStream << "root=" << (uint64_t)this << endl;
  graphStream << (uint64_t)this << " [label=\"" << guess << " - " << possibleSolutions.size()
              << "\",shape=circle,color=red]" << endl;
  dumpChildren(graphStream);
}

template <typename StrategyConfig>
void Strategy<StrategyConfig>::dump(ofstream &graphStream) {
  if (!possibleSolutions.empty()) {
    graphStream << (uint64_t)this << " [label=\"" << guess << " - " << possibleSolutions.size() << "\"]" << endl;
  } else {
    graphStream << (uint64_t)this << " [label=\"" << guess << "\",fontcolor=green,style=bold]" << endl;
  }
  dumpChildren(graphStream);
}

template <typename StrategyConfig>
void Strategy<StrategyConfig>::dumpChildren(ofstream &graphStream) {
  for (const auto &m : nextMoves) {
    m.second->dump(graphStream);
    graphStream << (uint64_t)this << " -> " << (uint64_t)(m.second.get()) << " [label=\"" << m.first << "\"]" << endl;
  }
}
