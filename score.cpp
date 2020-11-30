// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "score.hpp"

#include <iomanip>
#include <unordered_map>

using namespace std;

// Pretty printing for streams
ostream &Score::dump(ostream &stream) const {
  ios state(nullptr);
  state.copyfmt(stream);
  stream << hex << setfill('0') << setw(2) << (uint32_t)result;
  stream.copyfmt(state);
  return stream;
}

ostream &operator<<(ostream &stream, const Score &r) { return r.dump(stream); }

// So Scores can be keys in hash tables.
size_t hash<Score>::operator()(const Score &s) const { return hash<int>()(s.result); }
