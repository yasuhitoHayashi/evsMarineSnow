#include "particle_tracking.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

using EventTuple = std::tuple<int, int, std::int64_t>;
using EventList = std::vector<EventTuple>;

constexpr int kIsolationCellSizePx = 2;
constexpr std::int64_t kIsolationCellSizeUs = 500;

// --- CSV input parsing ---
struct CsvEventRow {
  int x = 0;
  int y = 0;
  int pol = 1;
  std::int64_t t = 0;
};

static bool parse_int(std::string_view tok, int &out) {
  int value = 0;
  const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), value);
  if (res.ec != std::errc() || res.ptr != tok.data() + tok.size())
    return false;
  out = value;
  return true;
}

static bool parse_i64(std::string_view tok, std::int64_t &out) {
  std::int64_t value = 0;
  const auto res =
      std::from_chars(tok.data(), tok.data() + tok.size(), value);
  if (res.ec != std::errc() || res.ptr != tok.data() + tok.size())
    return false;
  out = value;
  return true;
}

static bool split_event_csv_fields(std::string_view line,
                                   std::string_view (&fields)[4]) {
  if (line.empty())
    return false;

  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t pos = line.find(',');
    if (pos == std::string_view::npos)
      return false;
    fields[i] = line.substr(0, pos);
    line.remove_prefix(pos + 1);
  }

  if (line.find(',') != std::string_view::npos)
    return false;
  fields[3] = line;
  if (!fields[3].empty() && fields[3].back() == '\r')
    fields[3].remove_suffix(1);
  return true;
}

static bool parse_event_csv_row(const std::string &line, CsvEventRow &row) {
  std::string_view tokens[4];
  if (!split_event_csv_fields(line, tokens))
    return false;

  if (!parse_int(tokens[0], row.x))
    return false;
  if (!parse_int(tokens[1], row.y))
    return false;
  if (!parse_int(tokens[2], row.pol))
    return false;
  if (!parse_i64(tokens[3], row.t))
    return false;

  return true;
}

// --- output as ndjson ---
static void write_track_ndjson(std::ostream &out, const ParticleResult &track) {
  out << "{";
  out << "\"particle_id\":" << track.particle_id << ",";
  out << "\"events\":[";
  for (std::size_t i = 0; i < track.events.size(); ++i) {
    const auto &ev = track.events[i];
    out << "[" << std::get<0>(ev) << "," << std::get<1>(ev) << ","
        << std::get<2>(ev) << "]";
    if (i + 1 < track.events.size())
      out << ",";
  }
  out << "]}";
  out << "\n";
}

// --- remove isolated events ---
struct IsolationFilterOptions {
  bool enabled = false;
  int radius_px = 2;
  std::int64_t window_us = 1000;
  int min_neighbors = 1;
};

struct SpaceTimeCell {
  int x_bin;
  int y_bin;
  std::int64_t t_bin;
  bool operator==(const SpaceTimeCell &o) const noexcept {
    return x_bin == o.x_bin && y_bin == o.y_bin && t_bin == o.t_bin;
  }
};

struct SpaceTimeCellHash {
  std::size_t operator()(const SpaceTimeCell &cell) const noexcept {
    std::size_t h = 1469598103934665603ull;
    auto mix = [&](std::size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(static_cast<std::size_t>(static_cast<std::uint32_t>(cell.x_bin)));
    mix(static_cast<std::size_t>(static_cast<std::uint32_t>(cell.y_bin)));
    mix(static_cast<std::size_t>(static_cast<std::uint64_t>(cell.t_bin)));
    return h;
  }
};

using SpaceTimeGrid =
    std::unordered_map<SpaceTimeCell, std::vector<std::size_t>,
                       SpaceTimeCellHash>;

static int divide_rounding_up(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

static std::int64_t divide_rounding_up(std::int64_t numerator,
                                       std::int64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

static EventList keep_events_with_neighbors(const EventList &events,
                                            int neighbor_radius_px,
                                            std::int64_t neighbor_window_us,
                                            int min_neighbors) {
  if (events.empty())
    return {};

  const int neighbor_radius2 = neighbor_radius_px * neighbor_radius_px;
  const int x_cell_radius =
      divide_rounding_up(neighbor_radius_px, kIsolationCellSizePx);
  const int y_cell_radius = x_cell_radius;
  const std::int64_t t_cell_radius =
      divide_rounding_up(neighbor_window_us, kIsolationCellSizeUs);

  SpaceTimeGrid grid;

  auto to_cell = [&](int x, int y, std::int64_t t_us) -> SpaceTimeCell {
    const int x_bin = x / kIsolationCellSizePx;
    const int y_bin = y / kIsolationCellSizePx;
    const std::int64_t t_bin = t_us / kIsolationCellSizeUs;
    return SpaceTimeCell{x_bin, y_bin, t_bin};
  };

  for (std::size_t i = 0; i < events.size(); ++i) {
    const int x = std::get<0>(events[i]);
    const int y = std::get<1>(events[i]);
    const std::int64_t t = std::get<2>(events[i]);
    grid[to_cell(x, y, t)].push_back(i);
  }

  auto has_min_neighbors = [&](std::size_t i) -> bool {
    const int x0 = std::get<0>(events[i]);
    const int y0 = std::get<1>(events[i]);
    const std::int64_t t0 = std::get<2>(events[i]);

    const SpaceTimeCell center_cell = to_cell(x0, y0, t0);

    int neighbor_count = 0;
    for (std::int64_t dt_bin = -t_cell_radius; dt_bin <= t_cell_radius;
         ++dt_bin) {
      for (int dy_bin = -y_cell_radius; dy_bin <= y_cell_radius; ++dy_bin) {
        for (int dx_bin = -x_cell_radius; dx_bin <= x_cell_radius; ++dx_bin) {
          const SpaceTimeCell neighbor_cell{center_cell.x_bin + dx_bin,
                                           center_cell.y_bin + dy_bin,
                                           center_cell.t_bin + dt_bin};
          auto it = grid.find(neighbor_cell);
          if (it == grid.end())
            continue;
          const auto &idxs = it->second;

          for (std::size_t j : idxs) {
            if (j == i)
              continue;
            const int x1 = std::get<0>(events[j]);
            const int y1 = std::get<1>(events[j]);
            const std::int64_t t1 = std::get<2>(events[j]);

            const int dx = x1 - x0;
            const int dy = y1 - y0;
            const std::int64_t dt = (t1 >= t0) ? (t1 - t0) : (t0 - t1);
            if (dx * dx + dy * dy <= neighbor_radius2 &&
                dt <= neighbor_window_us) {
              ++neighbor_count;
              if (neighbor_count >= min_neighbors)
                return true;
            }
          }
        }
      }
    }

    return false;
  };

  EventList kept_events;
  kept_events.reserve(events.size());
  for (std::size_t i = 0; i < events.size(); ++i) {
    if (has_min_neighbors(i))
      kept_events.push_back(events[i]);
  }

  return kept_events;
}

static void remove_isolated_events_if_enabled(const IsolationFilterOptions &filter,
                                              EventList &events) {
  if (!filter.enabled)
    return;

  const std::size_t before_n = events.size();
  std::cerr << "Prefilter (isolated-event removal): enabled\n";
  std::cerr << "  cell_px=" << kIsolationCellSizePx
            << " cell_us=" << kIsolationCellSizeUs
            << " r_px=" << filter.radius_px
            << " w_us=" << filter.window_us
            << " k=" << filter.min_neighbors << "\n";

  auto filtered = keep_events_with_neighbors(
      events, filter.radius_px, filter.window_us, filter.min_neighbors);
  const std::size_t after_n = filtered.size();
  std::cerr << "  events: " << before_n << " -> " << after_n << " (kept "
            << (before_n ? (100.0 * double(after_n) / double(before_n)) : 0.0)
            << "%)\n";
  events.swap(filtered);
}

// --- parameters for tracking ---
struct CliOptions {
  double seconds = 0.0;

  double sigma_x = 2.0;
  double sigma_t = 10000.0;
  int m_threshold = 8;

  std::int64_t recent_us = 1000;
  std::int64_t retire_us = 10000;

  IsolationFilterOptions isolation;
};

// --- CLI ---
static CliOptions parse_cli_options(int argc, char **argv) {
  CliOptions opt;

  auto to_i = [](const std::string &s) { return std::stoi(s); };
  auto to_d = [](const std::string &s) { return std::stod(s); };

  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    const auto p = a.find('=');
    const std::string option_name =
        (p == std::string::npos) ? a : a.substr(0, p);
    const std::string option_value =
        (p == std::string::npos) ? "" : a.substr(p + 1);

    if (option_name == "--seconds")
      opt.seconds = to_d(option_value);
    else if (option_name == "--sigma_x")
      opt.sigma_x = to_d(option_value);
    else if (option_name == "--sigma_t")
      opt.sigma_t = to_d(option_value);
    else if (option_name == "--m_threshold")
      opt.m_threshold = to_i(option_value);
    else if (option_name == "--recent_us")
      opt.recent_us =
          static_cast<std::int64_t>(std::llround(to_d(option_value)));
    else if (option_name == "--retire_us")
      opt.retire_us =
          static_cast<std::int64_t>(std::llround(to_d(option_value)));
    else if (option_name == "--iso_enable")
      opt.isolation.enabled = (to_i(option_value) != 0);
    else if (option_name == "--iso_r_px")
      opt.isolation.radius_px = to_i(option_value);
    else if (option_name == "--iso_w_us")
      opt.isolation.window_us =
          static_cast<std::int64_t>(std::llround(to_d(option_value)));
    else if (option_name == "--iso_k")
      opt.isolation.min_neighbors = to_i(option_value);
  }

  return opt;
}

// --- event selection ---
static EventList read_windowed_events(std::ifstream &fin,
                                      const CliOptions &opt) {
  std::string line;

  EventList events;

  bool has_t0 = false;
  std::int64_t win_start = 0;
  std::int64_t win_end = 0;

  const std::int64_t duration_us =
      static_cast<std::int64_t>(std::llround(opt.seconds * 1e6));

  while (std::getline(fin, line)) {
    CsvEventRow row;
    if (!parse_event_csv_row(line, row))
      continue;
    if (row.pol != 1)
      continue;

    if (!has_t0) {
      has_t0 = true;
      win_start = row.t;
      if (duration_us > 0)
        win_end = win_start + duration_us;
    }

    if (duration_us > 0) {
      if (row.t >= win_end)
        break;
    }

    events.emplace_back(row.x, row.y, row.t);
  }

  std::cerr << "Processed events (windowed): " << events.size() << "\n";

  return events;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "ERROR: expected input.csv and output.ndjson.\n";
    return 1;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];

  CliOptions opt = parse_cli_options(argc, argv);

  std::ifstream fin(in_path);
  if (!fin) {
    std::cerr << "ERROR: cannot open input: " << in_path << "\n";
    return 2;
  }
  std::ofstream fout(out_path);
  if (!fout) {
    std::cerr << "ERROR: cannot open output: " << out_path << "\n";
    return 3;
  }
  auto events = read_windowed_events(fin, opt);
  if (events.empty()) {
    std::cerr << "No events after filtering/windowing.\n";
    return 0;
  }

  remove_isolated_events_if_enabled(opt.isolation, events);
  if (events.empty()) {
    std::cerr << "No events after isolated-event prefilter.\n";
    return 0;
  }

  const auto tracks =
      track_particles(events, opt.sigma_x, opt.sigma_t, opt.m_threshold,
                      opt.recent_us, opt.retire_us);

  for (const auto &track : tracks) {
    write_track_ndjson(fout, track);
  }

  std::cerr << "Tracks: " << tracks.size() << "\n";

  return 0;
}
