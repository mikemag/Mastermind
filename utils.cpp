// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.hpp"

#include <algorithm>
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
  if (runs.empty()) {
    cout << "No stats to write" << endl;
    return;
  }

  cout << "Writing all game stats to " << filename << endl;

  // Common data that we want to show up first, to make the files easier to read quickly.
  vector<string> common = {
      "Pin Count", "Color Count", "Strategy", "GPU Mode", "Initial Guess", "Average Turns", "Max Turns", "Elapsed (s)",
  };

  // Some runs may include stats that others don't, so aggregate the headers from them all so every row as the same
  // number of columns in the same order.
  set<string> headers;
  for (const auto &run : runs) {
    for (const auto &s : run) {
      if (find(begin(common), end(common), s.first) != common.end()) continue;
      headers.insert(s.first);
    }
  }

  ofstream ss(filename);

  // Write header row first
  auto cit = common.begin();
  ss << *cit;
  for (++cit; cit != common.end(); ++cit) {
    ss << "," << *cit;
  }
  for (const auto &h : headers) {
    ss << "," << h;
  }
  for (const auto &a : all) {
    ss << "," << a.first;
  }
  ss << endl;

  // One row per game
  for (auto &run : runs) {
    cit = common.begin();
    ss << run[*cit];
    for (++cit; cit != common.end(); ++cit) {
      ss << "," << run[*cit];
    }
    for (const auto &h : headers) {
      ss << "," << run[h];
    }
    for (const auto &a : all) {
      ss << "," << a.second;
    }
    ss << endl;
  }

  ss.close();
}
