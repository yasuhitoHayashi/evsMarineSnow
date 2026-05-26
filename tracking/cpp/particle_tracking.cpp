#include "particle_tracking.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>

struct Ev {
  int x;
  int y;
  std::int64_t t;
};

struct Gate {
  double inv_sx2;
  double inv_st2;
  double gate2;
};

class Particle {
public:
  Particle(int id, int x, int y, std::int64_t time_us,
           std::int64_t recent_win_us);

  void add_event(int x, int y, std::int64_t time_us);
  void merge_from(const Particle &other);
  bool is_active_final(int m_threshold) const;
  ParticleResult get_result() const;

  int particle_id;
  std::int64_t recent_window_us;
  std::deque<Ev> events;
  std::deque<Ev> recent_events;
};

// --- particle event history ---
static std::deque<Ev> merge_events_by_time(const std::deque<Ev> &a,
                                           const std::deque<Ev> &b) {
  std::deque<Ev> out;
  auto ia = a.begin();
  auto ib = b.begin();

  while (ia != a.end() && ib != b.end()) {
    if (ia->t <= ib->t) {
      out.push_back(*ia);
      ++ia;
    } else {
      out.push_back(*ib);
      ++ib;
    }
  }
  while (ia != a.end()) {
    out.push_back(*ia);
    ++ia;
  }
  while (ib != b.end()) {
    out.push_back(*ib);
    ++ib;
  }
  return out;
}

Particle::Particle(int id, int x, int y, std::int64_t time_us,
                   std::int64_t recent_win_us)
    : particle_id(id), recent_window_us(recent_win_us) {
  events.emplace_back(Ev{x, y, time_us});
  recent_events.emplace_back(Ev{x, y, time_us});
}

void Particle::add_event(int x, int y, std::int64_t time_us) {
  events.emplace_back(Ev{x, y, time_us});
  recent_events.emplace_back(Ev{x, y, time_us});

  const std::int64_t cutoff = time_us - recent_window_us;
  while (!recent_events.empty() && recent_events.front().t < cutoff) {
    recent_events.pop_front();
  }
}

void Particle::merge_from(const Particle &other) {
  events = merge_events_by_time(events, other.events);

  // Rebuild the search window after merging two time-ordered histories.
  recent_events.clear();
  if (!events.empty()) {
    const std::int64_t t_last = events.back().t;
    const std::int64_t cutoff = t_last - recent_window_us;
    for (auto it = events.rbegin(); it != events.rend(); ++it) {
      if (it->t < cutoff)
        break;
      recent_events.push_front(*it);
    }
  }
}

bool Particle::is_active_final(int m_threshold) const {
  return static_cast<int>(events.size()) >= m_threshold;
}

ParticleResult Particle::get_result() const {
  ParticleResult r;
  r.particle_id = particle_id;
  r.events.reserve(events.size());
  for (const auto &e : events) {
    r.events.emplace_back(e.x, e.y, e.t);
  }
  return r;
}

// --- gate test ---
Gate make_gate(double sx, double st) {
  if (!(sx > 0.0 && st > 0.0)) {
    throw std::invalid_argument("sigma_x and sigma_t must be > 0");
  }
  constexpr double CHI2_DF3_P95 = 7.81472; // 95% quantile for 3 dimensions.
  Gate g;
  g.inv_sx2 = 1.0 / (sx * sx);
  g.inv_st2 = 1.0 / (st * st);
  g.gate2 = CHI2_DF3_P95;
  return g;
}

static inline double gate_d2(const Gate &g, int x1, int y1, std::int64_t t1,
                             int x2, int y2, std::int64_t t2) {
  const double dx = static_cast<double>(x1) - static_cast<double>(x2);
  const double dy = static_cast<double>(y1) - static_cast<double>(y2);
  const double dt = static_cast<double>(t1) - static_cast<double>(t2);
  return (dx * dx + dy * dy) * g.inv_sx2 + (dt * dt) * g.inv_st2;
}

bool within_gate(const Gate &g, int x1, int y1, std::int64_t t1, int x2, int y2,
                 std::int64_t t2) {
  return gate_d2(g, x1, y1, t1, x2, y2, t2) <= g.gate2;
}

// --- particle tracking ---
std::vector<ParticleResult>
track_particles(const std::vector<std::tuple<int, int, std::int64_t>> &data,
                double sigma_x, double sigma_t, int m_threshold,
                std::int64_t recent_window_us, std::int64_t retire_window_us) {
  if (!(recent_window_us > 0 && retire_window_us > 0)) {
    throw std::invalid_argument(
        "recent_window_us and retire_window_us must be > 0");
  }

  std::vector<Ev> evs;
  evs.reserve(data.size());
  for (const auto &e : data) {
    evs.push_back(Ev{std::get<0>(e), std::get<1>(e), std::get<2>(e)});
  }
  std::stable_sort(evs.begin(), evs.end(),
                   [](const Ev &a, const Ev &b) { return a.t < b.t; });

  const Gate gate = make_gate(sigma_x, sigma_t);

  std::vector<Particle> particles;
  particles.reserve(1024);
  int next_id = 0;

  std::vector<ParticleResult> out;
  out.reserve(1024);

  auto hit_particle = [&](const Particle &p, int x, int y,
                          std::int64_t t) -> bool {
    for (const auto &re : p.recent_events) {
      if (within_gate(gate, x, y, t, re.x, re.y, re.t))
        return true;
    }
    return false;
  };

  auto retire_emit = [&](std::int64_t current_time) {
    const std::int64_t cutoff = current_time - retire_window_us;
    for (std::size_t i = 0; i < particles.size();) {
      const auto &p = particles[i];
      if (p.events.back().t < cutoff) {
        if (p.is_active_final(m_threshold))
          out.push_back(p.get_result());
        particles.erase(particles.begin() + static_cast<std::ptrdiff_t>(i));
      } else {
        ++i;
      }
    }
  };

  auto find_candidates = [&](int x, int y, std::int64_t t) {
    std::vector<int> cands;
    cands.reserve(8);
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
      if (hit_particle(particles[i], x, y, t))
        cands.push_back(i);
    }
    return cands;
  };

  auto merge_candidates_into_keep = [&](std::vector<int> &cands, int keep) {
    std::sort(cands.begin(), cands.end());
    for (int k = static_cast<int>(cands.size()) - 1; k >= 0; --k) {
      const int idx = cands[k];
      if (idx == keep)
        continue;
      particles[keep].merge_from(particles[idx]);
      particles.erase(particles.begin() + static_cast<std::ptrdiff_t>(idx));
    }
  };

  for (const auto &ev : evs) {
    const int x = ev.x;
    const int y = ev.y;
    const std::int64_t time = ev.t;

    retire_emit(time);

    auto cands = find_candidates(x, y, time);

    if (cands.empty()) {
      particles.emplace_back(++next_id, x, y, time, recent_window_us);
    } else if (cands.size() == 1) {
      particles[cands[0]].add_event(x, y, time);
    } else {
      const int keep = *std::min_element(cands.begin(), cands.end());
      merge_candidates_into_keep(cands, keep);
      particles[keep].add_event(x, y, time);
    }
  }

  for (const auto &p : particles) {
    if (p.is_active_final(m_threshold))
      out.push_back(p.get_result());
  }

  return out;
}
