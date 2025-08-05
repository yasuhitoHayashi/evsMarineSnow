#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <deque>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

// Struct for results
using Centroid = std::tuple<float, double, double>;
struct ParticleResult {
    int particle_id;
    std::vector<Centroid> centroid_history;
    std::vector<std::tuple<int,int,float>> events;  // (x,y,time)
};

class Particle {
public:
    int particle_id;
    int mass;
    std::vector<std::tuple<int,int,float>> events;
    std::deque<std::tuple<int,int,float>> recent_events;
    std::vector<Centroid> centroid_history;

    Particle(int id, int x, int y, float time)
      : particle_id(id), mass(1)
    {
        events.emplace_back(x, y, time);
        recent_events.emplace_back(x, y, time);
        centroid_history.emplace_back(time, x, y);
    }

    void update_centroid(float time) {
        double sum_x = 0, sum_y = 0;
        for (auto const& e : recent_events) {
            sum_x += std::get<0>(e);
            sum_y += std::get<1>(e);
        }
        centroid_history.emplace_back(time,
            sum_x / recent_events.size(),
            sum_y / recent_events.size()
        );
    }

    void add_event(int x, int y, float time) {
        events.emplace_back(x, y, time);
        recent_events.emplace_back(x, y, time);
        mass++;
        float cutoff = time - 2000.0f;  // 2 ms pruning
        while (!recent_events.empty() &&
               std::get<2>(recent_events.front()) < cutoff) {
            recent_events.pop_front();
        }
        update_centroid(time);
    }

    // Fixed merge weighting
    void merge(const Particle& other) {
        for (auto const& e : other.events) {
            events.push_back(e);
            recent_events.push_back(e);
        }
        int old_mass = mass;
        mass += other.mass;
        auto [t0, x0, y0] = centroid_history.back();
        auto [t1, x1, y1] = other.centroid_history.back();
        double total = double(old_mass) + double(other.mass);
        double mx = (x0*old_mass + x1*other.mass) / total;
        double my = (y0*old_mass + y1*other.mass) / total;
        centroid_history.emplace_back(
            std::get<2>(recent_events.back()), mx, my
        );
    }

    bool is_active(float current_time, int m_threshold) const {
        float last_t = std::get<2>(events.back());
        if (last_t < current_time - 2000.0f)
            return mass > m_threshold;
        return true;
    }
    bool is_active_final(int m_threshold) const {
        return mass > m_threshold;
    }

    ParticleResult get_result() const {
        return { particle_id, centroid_history, events };
    }
};

// Optimized top-hat overlap: precomputed inverses, early-exit
static inline bool elliptical_tophat_fast(int x1, int y1, float t1,
                                          int x2, int y2, float t2,
                                          double inv_sigma_x,
                                          double inv_sigma_t)
{
    double dt = (t1 - t2) * inv_sigma_t;
    double dt2 = dt * dt;
    if (dt2 > 1.0) return false;
    double dx = (x1 - x2) * inv_sigma_x;
    double dy = (y1 - y2) * inv_sigma_x;
    return (dx*dx + dy*dy + dt2) <= 1.0;
}

// トップハットによる重なり判定
// static bool elliptical_tophat(int x1, int y1, float t1,
//                               int x2, int y2, float t2,
//                               double sigma_x, double sigma_t)
// {
//     double dx = (x1 - x2) / sigma_x;
//     double dy = (y1 - y2) / sigma_x;  // 空間は同じ σₓ
//     double dt = (t1 - t2) / sigma_t;
//     return (dx*dx + dy*dy + dt*dt) <= 1.0;
// }

PYBIND11_MODULE(particle_tracking, m) {
    py::class_<ParticleResult>(m, "ParticleResult")
        .def_readonly("particle_id", &ParticleResult::particle_id)
        .def_readonly("centroid_history", &ParticleResult::centroid_history)
        .def_readonly("events", &ParticleResult::events);

    m.def("track_particles_cpp",
          [](const std::vector<std::tuple<int,int,float>>& data,
             double sigma_x, double sigma_t,
             int m_threshold)
          {
              if (data.empty())
                  throw std::invalid_argument("Input data is empty.");

              // precompute inverses
              double inv_sigma_x = 1.0 / sigma_x;
              double inv_sigma_t = 1.0 / sigma_t;

              std::vector<Particle> active, finished;
              int id_counter = 0;

              for (auto const& evt : data) {
                  int x, y; float t;
                  std::tie(x, y, t) = evt;

                  std::vector<size_t> overlap;
                  for (size_t i = 0; i < active.size(); ++i) {
                      for (auto const& re : active[i].recent_events) {
                          if (elliptical_tophat_fast(
                                x, y, t,
                                std::get<0>(re), std::get<1>(re), std::get<2>(re),
                                inv_sigma_x, inv_sigma_t))
                          {
                              overlap.push_back(i);
                              break;
                          }
                      }
                  }

                  if (overlap.empty()) {
                      active.emplace_back(++id_counter, x, y, t);
                  } else {
                      size_t base = overlap[0];
                      active[base].add_event(x, y, t);
                      if (overlap.size() > 1) {
                          std::vector<size_t> to_merge(
                              overlap.begin()+1, overlap.end());
                          std::sort(to_merge.begin(), to_merge.end(),
                                    std::greater<size_t>());
                          for (auto idx : to_merge) {
                              active[base].merge(active[idx]);
                              active.erase(active.begin() + idx);
                          }
                      }
                  }

                  // prune inactive
                  for (auto it = active.begin(); it != active.end();) {
                      if (!it->is_active(t, m_threshold)) {
                          finished.push_back(*it);
                          it = active.erase(it);
                      } else {
                          ++it;
                      }
                  }
              }

              // finalize remainder
              for (auto const& p : active) finished.push_back(p);
              finished.erase(
                  std::remove_if(finished.begin(), finished.end(),
                                 [&](Particle const& p){ return !p.is_active_final(m_threshold); }),
                  finished.end());

              std::vector<ParticleResult> results;
              results.reserve(finished.size());
              for (auto const& p : finished)
                  results.push_back(p.get_result());
              return results;
          },
          py::arg("data"),
          py::arg("sigma_x"),
          py::arg("sigma_t"),
          py::arg("m_threshold"),
          "Track particles with optimized pure top-hat association");
}
