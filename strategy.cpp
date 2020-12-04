// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <fstream>
#include <sstream>

#include "utils.hpp"

using namespace std;

// The core of the gameplay, this does one level of the search for a secret based on the algorithm implemented in
// various subclasses. Returns the number of moves needed to find the secret.
template <uint8_t p, uint8_t c, bool log>
uint32_t Strategy<p, c, log>::findSecret(Codeword<p, c> secret, int depth) {
  if (depth == 0) {
    if (log) {
      cout << "Starting search for secret " << secret << ", initial guess is " << guess << " with "
           << commaString(savedPossibleSolutions.size()) << " possibilities." << endl;
    }
  }

  // Play a guess and see how we did.
  Score r = secret.score(guess);
  rootData->scoreCounterCPU++;
  depth += 1;
  if (log) {
    cout << endl << "Tried guess " << guess << " against secret " << secret << " => " << r << endl;
  }
  if (r == Codeword<p, c>::winningScore) {
    if (log) {
      cout << "Solution found after " << depth << " moves." << endl;
    }
    return depth;
  }

  // Do we already have a move based on the result?
  shared_ptr<Strategy<p, c, log>> next = nextMoves[r];
  if (next) {
    if (log) {
      cout << "Following saved strategy for next move..." << endl;
    }
    return next->findSecret(secret, depth);
  }

  // Compute a new move given the result of our previous guess
  possibleSolutions = savedPossibleSolutions;
  removeImpossibleSolutions(r);

  // Figure out what our next guess should be
  Codeword<p, c> nextGuess;
  if (possibleSolutions.size() == 1) {
    nextGuess = possibleSolutions.front();
    possibleSolutions.clear();
    if (log) {
      cout << "Only remaining solution must be correct: " << nextGuess << endl;
    }
  } else if (enableTwoPSShortcut && possibleSolutions.size() == 2) {
    nextGuess = possibleSolutions.front();
    possibleSolutions.erase(begin(possibleSolutions));
    if (log) {
      cout << "Only two solutions remain, selecting the first one blindly: " << nextGuess << endl;
    }
    ++rootData->twoPSShortcuts;
  } else if (enableSmallPSShortcut && possibleSolutions.size() <= totalScores) {
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
template <uint8_t p, uint8_t c, bool log>
void Strategy<p, c, log>::removeImpossibleSolutions(Score r) {
  if (log) {
    cout << "Removing inconsistent possibilities... ";
  }
  rootData->scoreCounterCPU += possibleSolutions.size();
  possibleSolutions.erase(remove_if(possibleSolutions.begin(), possibleSolutions.end(),
                                    [&](Codeword<p, c> codeword) { return codeword.score(guess) != r; }),
                          possibleSolutions.end());
  if (log) {
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
// TODO: notes on speed improvements, selecting better guesses than some algs (prove it), tradeoff for the extra
// overhead.

template <uint8_t p, uint8_t c, bool log>
Codeword<p, c> Strategy<p, c, log>::shortcutSmallSets() {
  // TODO: skip this if totalCodewords is "too small", i.e., we won't save much work vs. the real thing.
  //  - need to measure and figure out what the breakeven point is.

  int subsetSizes[maxScoreSlots];
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
      if (log) {
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

template <uint8_t p, uint8_t c, bool l>
void Strategy<p, c, l>::printStats(std::chrono::duration<float, std::milli> elapsedMS) {
  cout << "Codeword comparisons: CPU = " << commaString(rootData->scoreCounterCPU)
       << ", GPU = " << commaString(rootData->scoreCounterGPU)
       << ", total = " << commaString(rootData->scoreCounterCPU + rootData->scoreCounterGPU) << endl;

  if (rootData->enableTwoPSMetrics) {
    cout << rootData->twoPSShortcuts << endl;
  }

  if (rootData->enableSmallPSMetrics) {
    cout << rootData->smallPSHighShortcuts << endl;
    cout << rootData->smallPSHighWasted << endl;
    cout << rootData->smallPSHighScores << endl;
    cout << rootData->smallPSInnerShortcuts << endl;
    cout << rootData->smallPSInnerWasted << endl;
    cout << rootData->smallPSInnerScoresSkipped << endl;
  }
}

template <uint8_t p, uint8_t c, bool l>
void Strategy<p, c, l>::recordStats(StatsRecorder &sr, std::chrono::duration<float, std::milli> elapsedMS) {
  sr.add("CPU Scores", rootData->scoreCounterCPU);
  sr.add("GPU Scores", rootData->scoreCounterGPU);

  if (rootData->enableTwoPSMetrics) {
    rootData->twoPSShortcuts.record(sr);
  }

  if (rootData->enableSmallPSMetrics) {
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
template <uint8_t pinCount, uint8_t colorCount, bool l>
void Strategy<pinCount, colorCount, l>::dump() {
  ostringstream fnStream;
  string algName = getName();
  replace(algName.begin(), algName.end(), ' ', '_');
  std::transform(algName.begin(), algName.end(), algName.begin(), ::tolower);
  fnStream << "mastermind_strategy_" << algName << "_" << (int)pinCount << "p" << (int)colorCount << "c.gv";
  string filename = fnStream.str();

  cout << "\nWriting strategy to " << filename << endl;
  ofstream graphStream(filename);
  graphStream << "digraph Mastermind_Strategy_" << getName() << "_" << (int)pinCount << "p" << (int)colorCount << "c";
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

template <uint8_t p, uint8_t c, bool l>
void Strategy<p, c, l>::dumpRoot(ofstream &graphStream) {
  graphStream << "root=" << (uint64_t)this << endl;
  graphStream << (uint64_t)this << " [label=\"" << guess << " - " << possibleSolutions.size()
              << "\",shape=circle,color=red]" << endl;
  dumpChildren(graphStream);
}

template <uint8_t p, uint8_t c, bool l>
void Strategy<p, c, l>::dump(ofstream &graphStream) {
  if (!possibleSolutions.empty()) {
    graphStream << (uint64_t)this << " [label=\"" << guess << " - " << possibleSolutions.size() << "\"]" << endl;
  } else {
    graphStream << (uint64_t)this << " [label=\"" << guess << "\",fontcolor=green,style=bold]" << endl;
  }
  dumpChildren(graphStream);
}

template <uint8_t p, uint8_t c, bool l>
void Strategy<p, c, l>::dumpChildren(ofstream &graphStream) {
  for (const auto &m : nextMoves) {
    m.second->dump(graphStream);
    graphStream << (uint64_t)this << " -> " << (uint64_t)(m.second.get()) << " [label=\"" << m.first << "\"]" << endl;
  }
}
