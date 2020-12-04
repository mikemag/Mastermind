// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iostream>

// Represents the result of scoring two Codewords.
//
// The two kinds of "hits" are packed into a single byte, 4-bits each. This effectively limits the number of pegs and
// colors to 15 each.
class Score {
 public:
  uint8_t result;

  constexpr Score() : result(0xFFu) {}
  constexpr Score(uint8_t b, uint8_t w) noexcept : result((b << 4u) | w) {}

  bool isInvalid() const { return result == 0xFFu; }

  bool operator==(const Score &other) const { return result == other.result; }
  bool operator!=(const Score &other) const { return !operator==(other); }

  std::ostream &dump(std::ostream &stream) const;
};

std::ostream &operator<<(std::ostream &stream, const Score &r);

template <>
struct std::hash<Score> {
  std::size_t operator()(const Score &s) const;
};
