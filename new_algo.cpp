//
// Created by Michael Magruder on 6/30/22.
//

#include "new_algo.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "codeword.hpp"

constexpr static int PIN_COUNT = 4;
constexpr static int COLOR_COUNT = 6;
using CodewordT = Codeword<PIN_COUNT, COLOR_COUNT>;
constexpr static uint32_t MAX_SCORE_SLOTS = (PIN_COUNT << 4u) + 1;

static CodewordT nextGuess(const vector<CodewordT>& possibleSolutions, const vector<CodewordT>& usedCodewords) {
  CodewordT bestGuess;
  size_t bestScore = 0;
  bool bestIsPossibleSolution = false;
  auto& allCodewords = CodewordT::getAllCodewords();
  int subsetSizes[MAX_SCORE_SLOTS];
  fill(begin(subsetSizes), end(subsetSizes), 0);
  for (const auto& g : allCodewords) {
    bool isPossibleSolution = false;
    for (const auto& ps : possibleSolutions) {
      Score r = g.score(ps);
      subsetSizes[r.result]++;
      if (r == CodewordT::WINNING_SCORE) {
        isPossibleSolution = true;  // Remember if this guess is in the set of possible solutions
      }
    }

    //    size_t score = computeSubsetScore();
    int largestSubsetSize = 0;  // Maximum number of codewords that could be retained by using this guess
    for (auto& s : subsetSizes) {
      if (s > largestSubsetSize) {
        largestSubsetSize = s;
      }
      s = 0;
    }

    // Invert largestSubsetSize, and return the minimum number of codewords that could be eliminated by using this guess
    int score = possibleSolutions.size() - largestSubsetSize;

    if (score > bestScore || (!bestIsPossibleSolution && isPossibleSolution && score == bestScore)) {
      if (find(usedCodewords.cbegin(), usedCodewords.cend(), g) != usedCodewords.end()) {
        continue;  // Ignore codewords we've already used
      }
      bestScore = score;
      bestGuess = g;
      bestIsPossibleSolution = isPossibleSolution;
    }
  }
  return bestGuess;
}

struct RegionIDLR {
  unsigned __int128 value = 0;
  uint32_t index;

  void append(const Score& s, int depth) {
    assert(depth < 16);
    value |= static_cast<unsigned __int128>(s.result) << ((16ull - depth) * 8ull);
  }

  bool isFinal() const {
    auto v = value;
    while (v != 0) {
      if ((v & 0xFF) == CodewordT::WINNING_SCORE.result) return true;
      v >>= 8;
    }
    return false;
  }

  std::ostream& dump(std::ostream& stream) const {
    std::ios state(nullptr);
    state.copyfmt(stream);
    stream << std::hex << std::setfill('0') << std::setw(16) << *(((uint64_t*)&value) + 1) << "-" << std::setw(16)
           << *((uint64_t*)&value);
    stream.copyfmt(state);
    return stream;
  }
};

struct RegionIDRL {
  unsigned __int128 value = 0;
  uint32_t index;

  void append(const Score& s, int depth) {
    assert(depth < 16);
    value = (value << 8) | s.result;
  }

  bool isFinal() const { return (value & 0xFF) == CodewordT::WINNING_SCORE.result; }

  std::ostream& dump(std::ostream& stream) const {
    std::ios state(nullptr);
    state.copyfmt(stream);
    stream << std::hex << std::setfill('0') << std::setw(16) << *(((uint64_t*)&value) + 1) << "-" << std::setw(16)
           << *((uint64_t*)&value);
    stream.copyfmt(state);
    return stream;
  }
};

using RegionID = RegionIDRL;

std::ostream& operator<<(std::ostream& stream, const RegionID& r) { return r.dump(stream); }

struct RegionRun {
  RegionID region;
  int start;
  int length;
};

// scoring all games in a region subdivides it, one new region per score.
// - sorting the region by score gets us runs of games we can apply the same guess to

// region id: [s1, s2, s3, s4, ...] <-- sort by that, stable to keep lexical ordering

// - start: all games get the same initial guess
//
// - while any games have guesses to play:
//   - score all games against their next guess, if any, which was given per-region
//     - append their score to their region id
//     - if no games have guesses, then we're done
//   - sort all games by region id
//     - this re-shuffles within each region only
//   - get start and run length for each region by id
//     - at the start there are 14, at the end there are likely 1296
//   - for reach region:
//     - find next guess using the region as PS, this is the next guess for all games in this region
//       - games with a win at the end of their region id get no new guess

// for GPU
//
// - start:
//   - next_guesses[0..n] = IG
//   - PS[0..n] = AC[0..n]
//   - region_id[0..n] = {}
//
// - while true:
//   - grid to score PS[0..n] w/ next_guesses[0..n] => updated region_id[0..n]
//     - s = PS[i].score(next_guesses[i])
//     - s is in a local per-thread
//     - append s to region_id[i]
//   - reduce scores to a single, non-winning score, if any
//   - if no non-winning scores, break, we're done
//
//   - grid to sort PS by region_id
//
//   - grid to reduce PS to a set of regions
//     - list of region_id, start index, and length
//
//   - grid per-region to find next guess and update next_guesses
//     - regions with a win at the end get no new guess, -1ish

// or, for the last two:
//
//   - grid over PS
//     - first thread in region kicks off the work for finding the next guess for that region
//       - first thread if rid_i != rid_(i-1)
//     - when done, shares the ng w/ all threads in the region and they update next_guesses[0..n]
//     - trying to avoid the device-wide reduction and extra kernel kickoff's per-region

void new_algo::run() {
  vector<CodewordT>& allCodewords = CodewordT ::getAllCodewords();

  // Starting case: all games, initial guess.
  //  vector<CodewordT> used{};
  vector<CodewordT> nextMoves(allCodewords.size(), 0x1122);
  vector<RegionID> regions(allCodewords.size());
  for (int i = 0; i < regions.size(); i++) regions[i].index = i;

  int depth = 0;
  size_t maxDepth = 0;
  size_t totalTurns = 0;

  // If no games have new moves, then we're done
  bool anyNewMoves = true;
  while (anyNewMoves) {
    depth++;
    printf("\n---------- depth = %d ----------\n\n", depth);

    // Score all games against their next guess, if any, which was given per-region
    for (auto& r : regions) {
      auto cwi = r.index;
      if (!nextMoves[cwi].isInvalid()) {
        auto s = allCodewords[cwi].score(nextMoves[cwi]);
        // Append the score to the region id
        r.append(s, depth);
        if (s == CodewordT::WINNING_SCORE) {
          maxDepth = max(maxDepth, (size_t)depth);
          totalTurns += depth;
        }
      }
    }

    // Sort all games by region id. This re-shuffles within each region only.
    std::stable_sort(regions.begin(), regions.end(),
                     [&](const RegionID& a, const RegionID& b) { return a.value < b.value; });

    // Get start and run length for each region by id.
    //   - at the start there are 14, at the end there are 1296
    vector<RegionRun> regionRuns{};
    regionRuns.push_back({regions[0], 0, 0});
    for (int i = 0; i < regions.size(); i++) {
      if (regions[i].value == regionRuns.back().region.value) {
        regionRuns.back().length++;
      } else {
        regionRuns.push_back({regions[i], i, 1});
      }
    }

    printf("%lu regions\n", regionRuns.size());

    // For reach region:
    //   - find next guess using the region as PS, this is the next guess for all games in this region
    //   - games already won (a win at the end of their region id) get no new guess
    anyNewMoves = false;
    for (auto& run : regionRuns) {
      cout << run.region << " -- " << run.start << ":" << run.length << " -- ";
      if (run.region.isFinal()) {
        cout << "win" << endl;
        for (int j = run.start; j < run.start + run.length; j++) {
          nextMoves[regions[j].index] = {};
        }
      } else {
        vector<CodewordT> ps;
        ps.reserve(run.length);
        for (int j = run.start; j < run.start + run.length; j++) {
          ps.push_back(allCodewords[regions[j].index]);
        }
        CodewordT ng = nextGuess(ps, {});  // mmmfixme: what if we need usedCodewords?
        cout << "ng = " << ng << endl;
        for (int j = run.start; j < run.start + run.length; j++) {
          nextMoves[regions[j].index] = ng;
        }
        anyNewMoves = true;
      }
    }
  }

  cout << "Max depth: " << maxDepth << endl;
  printf("Average number of turns was %.4f\n", (double)totalTurns / allCodewords.size());
}
