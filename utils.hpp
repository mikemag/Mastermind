// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <climits>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__CUDA_ARCH__)
#define CUDA_HOST_AND_DEVICE __device__ __host__
#else
#define CUDA_HOST_AND_DEVICE
#endif

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

// Round up to the next power of two
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type,
          typename = typename std::enable_if<std::is_unsigned<T>::value>::type>
constexpr T nextPowerOfTwo(T value, unsigned maxb = sizeof(T) * CHAR_BIT, unsigned curb = 1) {
  return maxb <= curb ? value : nextPowerOfTwo(((value - 1) | ((value - 1) >> curb)) + 1, maxb, curb << 1);
}

// Get basic info about the OS, hardware, machine, etc.
class OSInfo {
 public:
  std::map<std::string, std::string> info;

  OSInfo();

  std::ostream &dump(std::ostream &stream) const {
    for (const auto &a : info) {
      stream << a.first << ": " << a.second << std::endl;
    }
    return stream;
  }

  template <typename T>
  std::string macOSSysctlByName(const std::string &name);
};

template <bool enabled>
std::ostream &operator<<(std::ostream &stream, const OSInfo &r) {
  return r.dump(stream);
}

// Get basic info about the GPU, if present
class GPUInfo {
 public:
  std::map<std::string, std::string> info;

  GPUInfo();

  bool hasGPU() const { return hasGPU_; }

  std::ostream &dump(std::ostream &stream) const {
    for (const auto &a : info) {
      stream << a.first << ": " << a.second << std::endl;
    }
    return stream;
  }

 private:
  int _ConvertSMVer2Cores(int major, int minor);
  void dumpDeviceInfo();
  void loadDeviceInfo();

  bool hasGPU_ = false;
};

template <bool enabled>
std::ostream &operator<<(std::ostream &stream, const GPUInfo &r) {
  return r.dump(stream);
}

// A way to record various stats about a game so we can write a nice csv of results.
class StatsRecorder {
 public:
  std::vector<std::unordered_map<std::string, std::string>> runs;
  size_t currentRun = 0;
  std::map<std::string, std::string> all;
  OSInfo osInfo;
  GPUInfo gpuInfo;

  StatsRecorder() {
    for (const auto &a : osInfo.info) {
      all[a.first] = a.second;
    }

    for (const auto &a : gpuInfo.info) {
      all[a.first] = a.second;
    }
  }

  void newRun() {
    runs.emplace_back();
    currentRun = runs.size() - 1;
  }

  template <typename T>
  void add(const std::string &name, T value) {
    std::stringstream ss;
    ss << value;
    runs[currentRun][name] = ss.str();
  }

  void add(const std::string &name, const std::string &value) { runs[currentRun][name] = value; }
  void add(const std::string &name, const char *value) { runs[currentRun][name] = value; }

  template <typename T>
  void addAll(const std::string &name, T value) {
    std::stringstream ss;
    ss << value;
    all[name] = ss.str();
  }

  void writeStats(const std::string &filename);
};

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

  void record(StatsRecorder &sr) { sr.add(name, value); }
};

template <bool enabled>
std::ostream &operator<<(std::ostream &stream, const ExperimentCounter<enabled> &r) {
  return r.dump(stream);
}
