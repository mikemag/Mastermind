// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>

// CPU Reference Implementation
//
// This is a simple impl to serve as a reference for all the others. It's not optimized for speed. Hopefully it's clear.
//
// This plays all games at once, playing a turn on each game and forming a set of next guesses for the next turn.
// Scoring all games subdivides those games with the same score into disjoint regions. We form an id for each region
// made up of a growing list of scores, and sorting the list of games by region id groups the regions together. Then we
// can find a single best guess for each region and play that for all games in the region.
//
// Region ids are chosen to ensure that a stable sort keeps games within each region in their original lexical order.
// Many algorithms pick the first lexical game on ties.
//
// Pseudocode for the algorithm:
//   start: all games get the same initial guess
//
//   while any games have guesses to play:
//     score all games against their next guess, if any, which was given per-region
//       append their score to their region id
//       if no games have guesses, then we're done
//     stable sort all games by region id
//     get start and run length for each region by id
//     for reach region:
//       games with a win at the end of their region id get no new guess
//       otherwise, find next guess using the region itself as the possible solutions set PS

template <typename SolverConfig>
std::chrono::nanoseconds SolverReferenceImpl<SolverConfig>::playAllGames(uint32_t packedInitialGuess) {
  auto startTime = chrono::high_resolution_clock::now();

  vector<CodewordT>& allCodewords = CodewordT::getAllCodewords();

  // Starting case: all games get the same initial guess, region ids are empty
  vector<CodewordT> nextMoves(allCodewords.size(), packedInitialGuess);

  regionIDs = vector<RegionID>(allCodewords.size());
  for (int i = 0; i < regionIDs.size(); i++) regionIDs[i].index = i;

  int depth = 0;
  bool anyNewMoves = true;

  // If no games have new moves, then we're done
  while (anyNewMoves) {
    depth++;
    nextMovesList.push_back(nextMoves);

    if (SolverConfig::LOG) printf("\nDepth = %d\n", depth);

    // Score all games against their next guess, if any, which was given per-region. Append the score to the game's
    // region id.
    for (auto& regionID : regionIDs) {
      if (!regionID.isGameOver()) {
        auto s = allCodewords[regionID.index].score(nextMoves[regionID.index]);
        scoreCount++;
        regionID.append(s.result, depth);
        if (s == CodewordT::WINNING_SCORE) {
          this->maxDepth = max(this->maxDepth, (size_t)depth);
          this->totalTurns += depth;
        }
      }
    }

    // Sort all games by region id
    std::stable_sort(regionIDs.begin(), regionIDs.end(),
                     [&](const RegionID& a, const RegionID& b) { return a.value < b.value; });

    // Get start and run length for each region
    struct RegionBounds {
      RegionID regionID;
      int start;
      int length;
    };
    vector<RegionBounds> regions{};
    regions.push_back({regionIDs[0], 0, 0});
    for (int i = 0; i < regionIDs.size(); i++) {
      if (regionIDs[i].value == regions.back().regionID.value) {
        regions.back().length++;
      } else {
        regions.push_back({regionIDs[i], i, 1});
      }
    }

    if (SolverConfig::LOG) printf("Number of regions: %lu\n", regions.size());

    // For reach region:
    //   games with a win at the end of their region id get no new guess
    //   otherwise, find next guess using the region itself as the possible solutions set PS
    anyNewMoves = false;
    for (auto& region : regions) {
      if (region.regionID.isGameOver()) {
        // Not strictly necessary, but helps when debugging
        for (int j = region.start; j < region.start + region.length; j++) {
          nextMoves[regionIDs[j].index] = {};
        }
      } else {
        vector<CodewordT> ps;
        ps.reserve(region.length);
        for (int j = region.start; j < region.start + region.length; j++) {
          ps.push_back(allCodewords[regionIDs[j].index]);
        }
        // TODO: I don't recall exactly which algos need these
        vector<CodewordT> usedCodewords;
        for (auto& previousMoves : nextMovesList) {
          usedCodewords.push_back(previousMoves[region.regionID.index]);
        }

        CodewordT ng = nextGuess(ps, usedCodewords);

        for (int j = region.start; j < region.start + region.length; j++) {
          nextMoves[regionIDs[j].index] = ng;
        }
        anyNewMoves = true;
      }
    }

    if (depth == 16) {
      printf("\nMax depth reached, impl is broken!\n");
      break;
    }
  }
  auto endTime = chrono::high_resolution_clock::now();
  return endTime - startTime;
}

// These algorithms all rely on splitting the remaining possible guesses into groups or subsets based on their scores
// vs each other.
//
// The core of these comes from the method described by Knuth in [1], which subsets the possibilities by score,
// comparing all remaining codewords (i.e., not yet guessed) to all current possible guesses. This is O(n^2) in the
// number of total codewords, though the size of the possible solutions set does decrease drastically with each play. Of
// course, n = c^PIN_COUNT, so this is very, very expensive for larger games. Only the sizes of the subsets are
// retained; the actual codewords in each subset are unnecessary.
//
// Given these subset sizes, other algorithms can proceed to come up with a rank for each possible guess, and the guess
// with the maximum rank will be used, favoring guesses which are still in the possible solution set on ties.
//
// There's a decent summary of Knuth's overall algorithm on Wikipedia, too:
// https://en.wikipedia.org/wiki/Mastermind_(board_game)
//
// References:
// [1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.
// https://www.cs.uni.edu/~wallingf/teaching/cs3530/resources/knuth-mastermind.pdf
//
// [2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March
// 2013, arXiv:1305.1010 [cs.GT].
//
// [3] Barteld Kooi, Yet another mastermind Strategy. International Computer Games Association Journal,
// 28(1):13–20, 2005. https://www.researchgate.net/publication/30485793_Yet_another_Mastermind_strategy

template <typename SolverConfig>
typename SolverConfig::CodewordT SolverReferenceImpl<SolverConfig>::nextGuess(
    const vector<CodewordT>& possibleSolutions, const vector<CodewordT>& usedCodewords) {
  using ALGO = typename SolverConfig::ALGO;

  CodewordT bestGuess;
  size_t bestRank = 0;
  bool bestIsPossibleSolution = false;
  auto& allCodewords = CodewordT::getAllCodewords();
  int subsetSizes[SolverConfig::MAX_SCORE_SLOTS];
  fill(begin(subsetSizes), end(subsetSizes), 0);

  for (const auto& g : allCodewords) {
    bool isPossibleSolution = false;
    for (const auto& ps : possibleSolutions) {
      Score r = g.score(ps);
      ALGO::accumulateSubsetSize(subsetSizes[r.result]);
      if (r == CodewordT::WINNING_SCORE) {
        isPossibleSolution = true;  // Remember if this guess is in the set of possible solutions
      }
    }
    scoreCount += possibleSolutions.size();

    typename ALGO::RankingAccumulatorType rankingAccumulator{};
    for (auto& s : subsetSizes) {
      if (s > 0) {
        ALGO::accumulateRanking(rankingAccumulator, s, possibleSolutions.size());
      }
      s = 0;
    }

    uint32_t rank = ALGO::computeRank(rankingAccumulator, possibleSolutions.size());

    if (rank > bestRank || (!bestIsPossibleSolution && isPossibleSolution && rank == bestRank)) {
      if (find(usedCodewords.cbegin(), usedCodewords.cend(), g) != usedCodewords.end()) {
        printf("Discarding used codeword!\n");
        continue;  // Ignore codewords we've already used
      }
      bestRank = rank;
      bestGuess = g;
      bestIsPossibleSolution = isPossibleSolution;
    }
  }
  return bestGuess;
}

template <typename SolverConfig>
void SolverReferenceImpl<SolverConfig>::dump() {
  Solver::dump<SolverConfig, CodewordT>(regionIDs);
}

template <typename SolverConfig>
vector<uint32_t> SolverReferenceImpl<SolverConfig>::getGuessesForGame(uint32_t packedCodeword) {
  return Solver::getGuessesForGame<SolverReferenceImpl, SolverConfig, CodewordT>(packedCodeword, regionIDs);
}
