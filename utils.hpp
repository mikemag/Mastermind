// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <unordered_map>
#include <vector>

// This embodies my love/hate relationship with C++. Simple shit just isn't simple :(
class comma_numpunct : public std::numpunct<char> {
 protected:
  virtual char do_thousands_sep() const { return ','; }
  virtual std::string do_grouping() const { return "\03"; }
};

template <typename T>
std::string commaString(T i) {
  std::locale comma_locale(std::locale(), new comma_numpunct());
  std::stringstream ss;
  ss.imbue(comma_locale);
  ss << std::setprecision(2) << std::fixed << i;
  return ss.str();
}

std::string commaString(float v);

// A constexpr pow()
template <typename T>
constexpr T constPow(T num, T pow) {
  return pow == 0 ? 1 : num * constPow(num, pow - 1);
}

// Counters for various experiments, no overhead if not enabled.
template <bool enabled>
class ExperimentCounter {
  uint64_t value = 0;
  const char *name;

 public:
  explicit ExperimentCounter(const char *name) : name(name) {}

  ExperimentCounter<enabled> &operator++() {
    if (enabled) {
      ++value;
    }
    return *this;
  }

  ExperimentCounter<enabled> &operator+=(int64_t rhs) {
    if (enabled) {
      value += rhs;
    }
    return *this;
  }

  std::ostream &dump(std::ostream &stream) const {
    stream << name << ": ";
    if (enabled) {
      stream << commaString(value);
    } else {
      stream << "counter disabled";
    }
    return stream;
  }
};

template <bool enabled>
std::ostream &operator<<(std::ostream &stream, const ExperimentCounter<enabled> &r) {
  return r.dump(stream);
}

// A way to record various stats about a game so we can write a nice csv of results.
class StatsRecorder {
 public:
  std::vector<std::unordered_map<std::string, std::string>> stats;
  size_t currentRun = 0;

  void newRun() {
    stats.emplace_back();
    currentRun = stats.size() - 1;
  }

  template <typename T>
  void add(const std::string &name, T value) {
    std::stringstream ss;
    ss << value;
    stats[currentRun][name] = ss.str();
  }

  void add(const std::string &name, const std::string &value) { stats[currentRun][name] = value; }
  void add(const std::string &name, const char *value) { stats[currentRun][name] = value; }

  void writeStats(const std::string &filename);
};
