// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>

// CPU Implementation w/ Some Optimizations
//
// This is a version that runs on the CPU and is structured very close to the CUDA version. It's much faster than the
// reference impl, and includes the most common gameplay shortcuts.
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
std::chrono::nanoseconds SolverCPUFaster<SolverConfig>::playAllGames(uint32_t packedInitialGuess) {
  constexpr static bool LOG = SolverConfig::LOG;
  auto startTime = chrono::high_resolution_clock::now();

  constexpr static int MAX_SUPPORTED_TURNS = 16;
  vector<CodewordT>& allCodewords = CodewordT::getAllCodewords();

  // Starting case: region ids are empty
  regionIDs = vector<RegionIDT>(allCodewords.size());
  for (int i = 0; i < regionIDs.size(); i++) regionIDs[i].index = i;

  // Space for PS
  vector<CodewordT> ps;
  ps.reserve(allCodewords.size());

  // Space for a next moves vector for each possible turn
  nextMovesList.resize(MAX_SUPPORTED_TURNS, vector<CodewordT>(allCodewords.size()));
  auto nextMovesIter = nextMovesList.begin();
  std::fill(nextMovesIter->begin(), nextMovesIter->end(), CodewordT(packedInitialGuess));

  uint depth = 0;
  auto regionIDsEnd = regionIDs.end();  // The set of active games contracts as we go

  while (true) {
    depth++;
    if (LOG) printf("\nDepth = %d\n", depth);

    // Score all games against their next guess, if any, which was given per-region. Append the score to the game's
    // region id.
    for (auto rit = regionIDs.begin(); rit != regionIDsEnd; ++rit) {
      auto& regionID = *rit;
      auto s = allCodewords[regionID.index].score((*nextMovesIter)[regionID.index]);
      regionID.append(s.result, depth);
    }
    counters[find_counter(counterDescs, "Scores")] += regionIDsEnd - regionIDs.begin();

    // Advance to a fresh next moves vector
    ++nextMovesIter;

    // Push won games to the end and focus on the remaining games
    regionIDsEnd = std::partition(regionIDs.begin(), regionIDsEnd, [](const RegionIDT& r) { return !r.isGameOver(); });

    if (LOG) printf("Number of games left: %ld\n", regionIDsEnd - regionIDs.begin());

    // If no games need new moves, then we're done
    if (regionIDsEnd - regionIDs.begin() == 0) break;

    // Sort all games by region id
    std::stable_sort(regionIDs.begin(), regionIDsEnd,
                     [&](const RegionIDT& a, const RegionIDT& b) { return a.value < b.value; });

    // Get start and run length for each region
    struct RegionBounds {
      RegionIDT regionID;
      uint32_t start;
      uint32_t length;
    };
    vector<RegionBounds> regions{};
    regions.push_back({regionIDs[0], 0, 0});
    for (uint32_t i = 0; i < regionIDsEnd - regionIDs.begin(); i++) {
      if (regionIDs[i].value == regions.back().regionID.value) {
        regions.back().length++;
      } else {
        regions.push_back({regionIDs[i], i, 1});
      }
    }

    if (LOG) printf("Number of regions: %lu\n", regions.size());

    // For reach region, find next guess using the region itself as the possible solutions set PS
    for (auto& region : regions) {
      CodewordT ng;
      if (region.length <= 2) {
        ng = allCodewords[region.regionID.index];
      } else {
        ps.clear();
        for (int j = region.start; j < region.start + region.length; j++) {
          ps.push_back(allCodewords[regionIDs[j].index]);
        }

        bool fdoptSuccessful = false;
        if (ps.size() < SolverConfig::TOTAL_SCORES) {
          fdoptSuccessful = shortcutSmallSets(ps, ng);
          counters[find_counter(counterDescs, "Scores")] += ps.size() * ps.size();
        }

        if (!fdoptSuccessful) {
          vector<CodewordT> usedCodewords;
          for (auto& previousMoves : nextMovesList) {
            usedCodewords.push_back(previousMoves[region.regionID.index]);
          }

          if constexpr (SolverConfig::SYMOPT) {
            auto reducedAC = getReducedAC(allCodewords, ps, usedCodewords, depth);
            ng = nextGuess(reducedAC, ps, usedCodewords);
            assert(ng == nextGuess(allCodewords, ps, usedCodewords));
          } else {
            ng = nextGuess(allCodewords, ps, usedCodewords);
          }
        }
      }

      for (int j = region.start; j < region.start + region.length; j++) {
        (*nextMovesIter)[regionIDs[j].index] = ng;
      }
    }

    if (depth == MAX_SUPPORTED_TURNS) {
      printf("\nMax depth reached, impl is broken!\n");
      break;
    }
  }
  auto endTime = chrono::high_resolution_clock::now();

  // Post-process for stats
  for (int i = 0; i < regionIDs.size(); i++) {
    auto c = regionIDs[i].countMovesPacked();
    this->maxDepth = max<size_t>(this->maxDepth, c);
    this->totalTurns += c;
  }

  // TODO: tmp stats dumping
  //  cout << "Depth, Hits, HitsFromBelow, Size, PSSize, Zeros, Frees, ZeroCnt, FreeCnt" << endl;
  //  for (auto const& [key, val] : symACCache) {
  //    cout << val.depth << ", " << val.hitCount << ", " << val.hitFromBelow << ", " << val.reducedAC.size() << ", "
  //         << val.psSize << ", " << key.isZero << ", " << key.isFree << ", " << std::popcount(key.isZero) << ", "
  //         << std::popcount(key.isFree) << endl;
  //  }
  //  cout << "Set count: " << symACCache.size() << endl;

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
typename SolverConfig::CodewordT SolverCPUFaster<SolverConfig>::nextGuess(const vector<CodewordT>& allCodewords,
                                                                          const vector<CodewordT>& possibleSolutions,
                                                                          const vector<CodewordT>& usedCodewords) {
  using ALGO = typename SolverConfig::ALGO;

  CodewordT bestGuess;
  size_t bestRank = 0;
  bool bestIsPossibleSolution = false;
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
    counters[find_counter(counterDescs, "Scores")] += possibleSolutions.size();

    typename ALGO::RankingAccumulatorType rankingAccumulator{};
    int totalSubsets = 0;
    for (auto& s : subsetSizes) {
      if (s > 0) {
        ALGO::accumulateRanking(rankingAccumulator, s, possibleSolutions.size());
        totalSubsets++;
      }
      s = 0;
    }

    // Any fully discriminating guess is an instant winner. Part of the shortcutSmallSets opt below.
    if (totalSubsets == possibleSolutions.size()) {
      return g;
    }

    uint32_t rank = ALGO::computeRank(rankingAccumulator, possibleSolutions.size());

    if (rank > bestRank || (!bestIsPossibleSolution && isPossibleSolution && rank == bestRank)) {
      if (find(usedCodewords.cbegin(), usedCodewords.cend(), g) != usedCodewords.end()) {
        continue;  // Ignore codewords we've already used
      }
      bestRank = rank;
      bestGuess = g;
      bestIsPossibleSolution = isPossibleSolution;
    }
  }
  return bestGuess;
}

// Optimization from [2]: if the possible solution set is smaller than the number of possible scores, and if one
// codeword can fully discriminate all of the possible solutions (i.e., it produces a different score for each one),
// then play it right away since it will tell us the winner.
//
// This is an interesting shortcut. It doesn't change the results of any of the subsetting algorithms at all: average
// turns, max turns, max secret, and the full histograms all remain precisely the same. What does change is the number
// of scores computed, and the runtime.
template <typename SolverConfig>
bool SolverCPUFaster<SolverConfig>::shortcutSmallSets(const vector<CodewordT>& possibleSolutions,
                                                      CodewordT& nextGuess) {
  bool subsetSizes[SolverConfig::MAX_SCORE_SLOTS];
  for (const auto& psa : possibleSolutions) {
    fill(begin(subsetSizes), end(subsetSizes), 0);
    for (const auto& psb : possibleSolutions) {
      Score r = psa.score(psb);
      subsetSizes[r.result] = true;
    }
    int totalSubsets = 0;
    for (auto s : subsetSizes) {
      if (s) {
        totalSubsets++;
      }
    }
    if (totalSubsets == possibleSolutions.size()) {
      nextGuess = psa;
      return true;
    }
  }
  return false;
}

// Optimization from Ville[2], section 5.4: Symmetry.
//
// I've adopted his terminology of "free" and "zero" colors. Free colors are those which have never been played.
// Zero colors are those which we know cannot be part of the secret. Zero colors add no new information when played,
// and thus they're all equal. So we can replace all of them with the lexically first one. I.e., if we know that colors
// 1, 2, and 3 can't be part of the secret, then we can replace any 2 or 3 with 1, and play that codeword to get the
// same result.
//
// Likewise, free colors provide the same information when played, no matter their order. If we know 1, 2, and 3 are
// free, then 1123, 1132, 2213, 2231, 3312, and 3321 all give the same information. So we would wish to replace free
// colors as we see them with the lexically ordered free colors, i.e., transform every codeword in the example to 1123.
//
// Free colors are easy to identify for any region: simply look at the used codewords set and collect the colors from
// each codeword played.
//
// Zero colors are, conceptually, discovered as guesses are played and the scores are understood. Practically, the PS
// set holds all codewords which are consistent with every guess played so far, and we can simply look at the entire
// set and collect all colors present. Any color not present in PS is a zero color.

// Rather tha actually transform codewords into their symmetric representative, simply determine if a given word is
// in fact that representative. It's always in the set already.
template <typename SolverConfig>
bool SolverCPUFaster<SolverConfig>::isSymmetricRepresentitive(uint32_t cw, const vector<uint32_t>& zeros,
                                                              const vector<uint8_t>& frees, uint32_t isFree) const {
  // Any color beyond the first would be switched to the first, to be symmetric, so stop early if we find one.
  for (int i = 1; i < zeros.size(); i++) {
    uint32_t mask = 0xF << ((SolverConfig::PIN_COUNT - 1) * 4);
    while (mask != 0) {
      if ((cw & mask) == (zeros[i] & mask)) {
        return false;
      }
      mask >>= 4;
    }
  }

  if (!frees.empty()) {
    // We want to replace all free colors, in random order, with the free colors in order. Thus, anything out of place
    // isn't the lexically first of the symmetric set, so we can stop early.
    uint32_t nextSub = 0;
    uint32_t mask = 0xF << ((SolverConfig::PIN_COUNT - 1) * 4);
    int i = SolverConfig::PIN_COUNT - 1;
    while (mask != 0) {
      auto p = (cw & mask) >> (i * 4);
      if (isFree & (1 << p)) {
        if (p != frees[nextSub++]) {
          return false;
        }
        isFree &= ~(1 << p);
      }
      mask >>= 4;
      i--;
    }
  }

  return true;
}

// mmmfixme: tmp
int factorial(int n) {
  if (n == 0) return 1;
  return n * factorial(n - 1);
}

// Produce a reduced AC which only contains representative codewords given the zero and free colors in PS.
template <typename SolverConfig>
vector<typename SolverConfig::CodewordT> SolverCPUFaster<SolverConfig>::getReducedAC(
    const vector<CodewordT>& allCodewords, const vector<CodewordT>& possibleSolutions,
    const vector<CodewordT>& usedCodewords, uint depth) {
  // We can form the Zeros set by looking at which colors are present in PS. PS represents all codewords which are
  // consistent with the guesses so far, and therefore contains the colors which could still be in the final solution.
  // If a color is absent, that color will not be part of the solution, and is therefore part of the Zeros set.
  typename CodewordT::CT usedColors = 0;
  for (auto& cw : possibleSolutions) {
    usedColors |= cw.packedColors();
  }

  // We can form the Frees set by looking at all colors played to reach this PS.
  typename CodewordT::CT playedColors = 0;
  for (const auto& cw : usedCodewords) {
    playedColors |= cw.packedColors();
  }

  // The subsetting algorithms expect the Zeros set to be in lexicographical order, so transformed codewords are as
  // well.
  // TODO: I'm not really a fan of how I've chosen to represent these. I feel like there's improvements to be had here.
  vector<uint32_t> zeros = {};
  uint32_t isZero = 0;
  vector<uint8_t> frees = {};
  uint32_t isFree = 0;
  int color = 1;
  while (color <= SolverConfig::COLOR_COUNT) {
    if ((usedColors & 0xFF) == 0) {
      zeros.push_back(0x11111111 * color);
      isZero |= (1 << color);
    } else if ((playedColors & 0xFF) == 0) {
      frees.push_back(color);
      isFree |= (1 << color);
    }
    usedColors >>= 8;
    playedColors >>= 8;
    color++;
  }

  // Sets of size one are useless, as any transformation does nothing.
  if (zeros.size() == 1) {
    zeros = {};
    isZero = 0;
  }
  if (frees.size() == 1) {
    frees = {};
    isFree = 0;
  }
  if (zeros.empty() && frees.empty()) {
    return allCodewords;
  }

  // Caching these on playedColors and usedColors. Hit rate is reasonable, but not as high as you'd initially guess.
  SymACKey key{isZero, isFree};
  if (symACCache.contains(key)) {
    // TODO: tmp stats collections
    auto& t = symACCache[key];
    if (t.depth != depth) {
      t.hitFromBelow += 1;
    } else {
      t.hitCount += 1;
    }
    return t.reducedAC;
  }

  // The representative codeword we'll use is the lexically first one in each set of equal transformed codewords. That's
  // the codeword which doesn't change during transformation, and we can just use it straightaway and toss the rest. We
  // don't actually have to form any new codewords, try to sort and unique them, etc. (which, frankly, was my first
  // idea.)
  vector<CodewordT> reducedAC;
  std::copy_if(allCodewords.begin(), allCodewords.end(), std::back_inserter(reducedAC),
               [&](auto cw) { return isSymmetricRepresentitive(cw.packedCodeword(), zeros, frees, isFree); });

  // mmmfixme:
  //  - Figuring out how many representative codewords exist without actually forming them.
  //  - Math works, these match for all cases for 4p12c, 4p15c, 4p6c, 5p8c, 2p15c, 8p2c, and a bunch more I've tested.
  //  - Math seems sound. Need to verify a smidge more, then write it up well.
  //  - The goal is to know ahead of time, given Frees and Zeros, how many codewords will be in reducedAC.
  //    - With this, for the GPU version, I can effecitvely allocate regions of memory, sort regions togehter based
  //      not only on their size, but on their size x their reducedAC.
  //      - May also be reasonable to run a different kernel for small reducedAC x small PS.
  int a[16][16] = {};
  int g[16] = {};
  a[2][1] = 1;
  g[1] = 1;
  for (int n = 2; n <= SolverConfig::PIN_COUNT; n++) {
    for (int x = 2; x <= n + 1; x++) {
      auto px = x;
      if (px < frees.size()) {
        a[x][n] = a[x - 1][n - 1] + (px - 1) * a[x][n - 1];
      } else if (px == frees.size()) {
        a[x][n] = a[x - 1][n - 1] + px * a[x][n - 1];
      }
      g[n] += a[x][n];
    }
  }

  int availableColors = 0;
  if (!frees.empty() && !zeros.empty()) {
    availableColors = SolverConfig::COLOR_COUNT - frees.size() - zeros.size() + 1;
  } else {
    availableColors = std::max<int>(1, SolverConfig::COLOR_COUNT - frees.size() - zeros.size());
  }

  int FS = 0;
  for (int p = 1; p <= SolverConfig::PIN_COUNT; p++) {
    int Cpx = factorial(SolverConfig::PIN_COUNT) / (factorial(SolverConfig::PIN_COUNT - p) * factorial(p));
    int yp = Cpx * g[p];
    int pw = pow(availableColors, SolverConfig::PIN_COUNT - p);
    FS += yp * pw;
  }

  int total = 0;
  if (!frees.empty()) {
    total = pow(availableColors, SolverConfig::PIN_COUNT) + FS;
  } else {
    total = pow(SolverConfig::COLOR_COUNT - zeros.size() + 1, SolverConfig::PIN_COUNT);
  }
  if (total != reducedAC.size()) {
    cout << "OOPS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
  }

  // TODO: tmp stats added to key.
  symACCache[key] = {reducedAC, depth, possibleSolutions.size()};
  return reducedAC;
}

template <typename SolverConfig>
void SolverCPUFaster<SolverConfig>::dump() {
  Solver::dump<SolverConfig, CodewordT>(regionIDs);
}

template <typename SolverConfig>
vector<uint32_t> SolverCPUFaster<SolverConfig>::getGuessesForGame(uint32_t packedCodeword) {
  return Solver::getGuessesForGame<SolverCPUFaster, SolverConfig, CodewordT>(packedCodeword, regionIDs);
}
