// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <fstream>
#include <sstream>

// Dump the tree of moves over all games as a GraphViz graph. These are fun to render.
//
// This uses the regions to recover the decision tree for all the games, and the saved next moves vectors to gather
// which guesses were played at each turn.
template <typename SolverConfig, typename CodewordT, typename RegionID>
void Solver::dump(vector<RegionID>& regionIDs) {
  ostringstream fnStream;
  string algName = SolverConfig::ALGO::name;
  replace(algName.begin(), algName.end(), ' ', '_');
  std::transform(algName.begin(), algName.end(), algName.begin(), ::tolower);
  fnStream << "mastermind_strategy_" << algName << "_" << (int)SolverConfig::PIN_COUNT << "p"
           << (int)SolverConfig::COLOR_COUNT << "c.gv";
  string filename = fnStream.str();

  cout << "\nWriting strategy to " << filename << endl;
  ofstream graphStream(filename);
  graphStream << "digraph Mastermind_Strategy_" << SolverConfig::ALGO::name << "_" << (int)SolverConfig::PIN_COUNT
              << "p" << (int)SolverConfig::COLOR_COUNT << "c";
  graphStream << " {" << endl;
  graphStream << "size=\"40,40\"" << endl;  // Good size for jpgs
  graphStream << "overlap=true" << endl;    // scale is cool, but the result is unreadable
  graphStream << "ranksep=5" << endl;
  graphStream << "ordering=out" << endl;
  graphStream << "node [shape=plaintext]" << endl;

  // Sort the regionIDs so that score edges are ordered nicely in the graph
  std::sort(regionIDs.begin(), regionIDs.end(),
            [&](const RegionID& a, const RegionID& b) { return a.value < b.value; });

  uint32_t ig = SolverConfig::ALGO::template presetInitialGuess<SolverConfig::PIN_COUNT, SolverConfig::COLOR_COUNT>();
  auto igstr = "IG";

  graphStream << std::hex;
  graphStream << "root=" << igstr << endl;
  graphStream << igstr << " [label=\"" << ig << "\",shape=circle,color=red]" << endl;

  for (int level = 1; level < maxDepth; level++) {
    auto last = numeric_limits<unsigned __int128>::max();
    for (auto& r : regionIDs) {
      auto rp = r.regionPrefix(level);
      if (rp != last && rp != numeric_limits<unsigned __int128>::max()) {
        uint8_t score = rp & 0xFF;
        if (score != RegionID::WINNING_SCORE) {
          auto to = getPackedCodewordForRegion(level, r.index);

          auto fmtRP = [](int level, unsigned __int128 v) {
            stringstream ss;
            ss << "\"" << std::hex << std::setw(level * 2) << std::setfill('0') << (unsigned long long)v << "\"";
            return ss.str();
          };

          graphStream << fmtRP(level, rp) << " [label=\"" << to << "\",fontcolor=green,style=bold]" << endl;

          if (level == 1) {
            graphStream << igstr;
          } else {
            auto prp = rp >> 8;
            graphStream << fmtRP(level - 1, prp);
          }
          graphStream << " -> " << fmtRP(level, rp) << " [label=\"" << std::setw(2) << std::setfill('0')
                      << (int)getStandardScore(score) << "\"]" << endl;
        } else {
          printf("Winner\n");
        }
        last = rp;
      }
    }
  }

  graphStream << "}" << endl;
  graphStream.close();
}
