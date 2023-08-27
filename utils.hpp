// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__CUDA_ARCH__)
#define CUDA_HOST_AND_DEVICE __device__ __host__
#else
#define CUDA_HOST_AND_DEVICE
#endif

using json = nlohmann::json;

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

std::string hexString(uint32_t v);

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
  std::map<std::string, json> info;

  OSInfo();

  std::ostream &dump(std::ostream &stream) const {
    for (const auto &a : info) {
      stream << a.first << ": " << a.second << std::endl;
    }
    return stream;
  }

  template <typename T>
  T macOSSysctlByName(const std::string &name);
};

template <bool enabled>
std::ostream &operator<<(std::ostream &stream, const OSInfo &r) {
  return r.dump(stream);
}

// Get basic info about the GPU, if present
class GPUInfo {
 public:
  std::map<std::string, json> info;

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

// A way to record various stats about a game, so we can write a nice json file of results.
class StatsRecorder {
 public:
  OSInfo osInfo;
  GPUInfo gpuInfo;

  explicit StatsRecorder(const std::string &filename);
  ~StatsRecorder();

  void newRun();

  void add(const std::string &name, const json &value) { run[name] = value; }

 private:
  std::ofstream js;
  std::string filename;
  std::unordered_map<std::string, json> run;
  static constexpr auto tmpFilename = "tmp_run_results.json";
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
