// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

// Holds all the constants we need to use any of our gameplay strategies.
template <uint8_t PIN_COUNT_, uint8_t COLOR_COUNT_, bool LOG_>
struct StrategyConfig {
  static constexpr uint8_t PIN_COUNT = PIN_COUNT_;
  static constexpr uint8_t COLOR_COUNT = COLOR_COUNT_;
  static constexpr bool LOG = LOG_;

  constexpr static int TOTAL_SCORES = (PIN_COUNT * (PIN_COUNT + 3)) / 2;
};

// Holds all the constants we need to use the subsetting gameplay strategies.
template <uint8_t PIN_COUNT_, uint8_t COLOR_COUNT_, bool LOG_, Algo ALGO_, typename SubsetSizeT_>
struct SubsettingStrategyConfig : StrategyConfig<PIN_COUNT_, COLOR_COUNT_, LOG_> {
  static constexpr Algo ALGO = ALGO_;
  using SubsetSizeT = SubsetSizeT_;
};
