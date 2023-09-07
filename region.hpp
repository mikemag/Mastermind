// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>

#include "utils.hpp"

// A simple region id which packs scores into a single 128-bit value, starting w/ the high order bits.
// The region id carries the index of the game, so we can reorder regions at will.

// A "region" represents a segment of the solution space for a game which remains given a path of guesses taken so far.
// In the beginning, it's all the possible solutions. After the initial guess the problem space is segmented into, say,
// a dozen different regions for a 4p6c game, each represented by the score of the initial guess vs. the solution.
//
// Regions get smaller as more guesses are played until individual games are won. They make a tree of moves and scores.
//
// Here, a maximum of 16 turns are allowed with scores packed into a single 128bit value. Packing is from most
// significant bits to least. Each score is a byte. Methods are provided to append scores, recognize games which have
// been won, and help with metrics and dumping game strategies after a run.
//
// nb: this has to run on device and host in GPU builds.
template <typename T, uint8_t WINNING_SCORE_>
struct RegionID {
  constexpr static auto WINNING_SCORE = WINNING_SCORE_;

  T value = 0;
  uint32_t index;

  CUDA_HOST_AND_DEVICE RegionID() : value(0), index(0) {}
  CUDA_HOST_AND_DEVICE RegionID(const RegionID& r) : value(r.value), index(r.index) {}

  // nb: depth starts at 1
  CUDA_HOST_AND_DEVICE void append(uint8_t s, int depth) {
    assert(depth < 16);
    value |= static_cast<T>(s) << (numeric_limits<T>::digits - (depth * CHAR_BIT));
  }

  CUDA_HOST_AND_DEVICE bool isGameOver() const {
    auto v = value;
    while (v != 0) {
      if ((v & 0xFF) == WINNING_SCORE) return true;
      v >>= 8;
    }
    return false;
  }

  CUDA_HOST_AND_DEVICE int countMovesPacked() const {
    auto v = value;
    int c = 0;
    while (v != 0) {
      c++;
      static constexpr auto highByteShift = numeric_limits<T>::digits - CHAR_BIT;
      if (((v & (static_cast<T>(0xFF) << highByteShift)) >> highByteShift) == WINNING_SCORE) break;
      v <<= 8;
    }
    return c;
  }

  // nb: depth starts at 1
  T regionPrefix(int depth) {
    auto shift = numeric_limits<T>::digits - (depth * CHAR_BIT);
    auto mask = (static_cast<T>(1) << shift) - 1;
    if ((value & mask) == 0) return numeric_limits<T>::max();
    return value >> shift;
  }

  uint8_t getScore(int depth) const {
    assert(depth < 16);
    return (value >> (numeric_limits<T>::digits - (depth * CHAR_BIT))) & 0xFF;
  }
};
