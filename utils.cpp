// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

#include <fstream>
#include <set>

using namespace std;

string commaString(float f) {
  locale comma_locale(locale(), new comma_numpunct());
  stringstream ss;
  ss.imbue(comma_locale);
  ss << setprecision(4) << fixed << f;
  return ss.str();
}

void StatsRecorder::writeStats(const std::string &filename) {
  // Some runs may include stats that others don't, so aggregate the headers from them all so every row as the same
  // number of columns in the same order.
  set<string> headers;
  for (const auto &run : stats) {
    for (const auto &s : run) {
      headers.insert(s.first);
    }
  }

  ofstream ss(filename);

  // Write header row first
  auto hit = headers.begin();
  ss << *hit;
  for (++hit; hit != headers.end(); ++hit) {
    ss << "," << *hit;
  }
  ss << endl;

  // One row per game
  for (auto &run : stats) {
    hit = headers.begin();
    ss << run[*hit];
    for (++hit; hit != headers.end(); ++hit) {
      ss << "," << run[*hit];
    }
    ss << endl;
  }

  ss.close();
}
