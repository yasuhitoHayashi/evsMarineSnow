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

// Struct for particle tracking result
typedef std::tuple<float, double, double> Centroid;
struct ParticleResult {
    int particle_id;
    std::vector<Centroid> centroid_history;
    std::vector<std::tuple<int, int, int, float>> events;
};

class Particle {
public:
    int particle_id;
    int mass;
    std::vector<std::tuple<int,int,int,float>> events;
    std::deque<std::tuple<int,int,int,float>> recent_events;
    std::vector<Centroid> centroid_history;

    Particle(int id, int x, int y, int polarity, float time)
      : particle_id(id), mass(1)
    {
        events.emplace_back(x, y, polarity, time);
        recent_events.emplace_back(x, y, polarity, time);
        centroid_history.emplace_back(time, x, y);
    }

    void update_centroid(float time) {
        double sum_x = 0.0, sum_y = 0.0;
        for (auto const& e : recent_events) {
            sum_x += std::get<0>(e);
            sum_y += std::get<1>(e);
        }
        double cx = sum_x / recent_events.size();
        double cy = sum_y / recent_events.size();
        centroid_history.emplace_back(time, cx, cy);
    }

    void add_event(int x, int y, int polarity, float time) {
        events.emplace_back(x, y, polarity, time);
        recent_events.emplace_back(x, y, polarity, time);
        mass++;
        // prune events older than 2000 us
        float cutoff = time - 2000.0f;
        while (!recent_events.empty() && std::get<3>(recent_events.front()) < cutoff) {
            recent_events.pop_front();
        }
        update_centroid(time);
    }

    // Fixed merge weighting: correct old_mass + other.mass
    void merge(const Particle& other) {
        // append all events
        for (auto const& e : other.events) {
            events.push_back(e);
            recent_events.push_back(e);
        }
        int old_mass = mass;
        mass += other.mass;
        Centroid c0 = centroid_history.back();
        double cx0 = std::get<1>(c0);
        double cy0 = std::get<2>(c0);
        Centroid c1 = other.centroid_history.back();
        double cx1 = std::get<1>(c1);
        double cy1 = std::get<2>(c1);
        double total = double(old_mass) + double(other.mass);
        double mx = (cx0 * old_mass + cx1 * other.mass) / total;
        double my = (cy0 * old_mass + cy1 * other.mass) / total;
        float tnew = std::get<3>(recent_events.back());
        centroid_history.emplace_back(tnew, mx, my);
    }

    bool is_active(float current_time, int m_threshold) const {
        float last_t = std::get<3>(events.back());
        if (last_t < current_time - 2000.0f) {
            return mass > m_threshold;
        }
        return true;
    }

    bool is_active_final(int m_threshold) const {
        return mass > m_threshold;
    }

    ParticleResult get_result() const {
        return { particle_id, centroid_history, events };
    }
};

// Gaussian kernel score for association
double gaussian_distance(int x1, int y1, float t1,
                         int x2, int y2, float t2,
                         double sigma_x, double sigma_t)
{
    double dx2 = double(x1 - x2) * (x1 - x2);
    double dy2 = double(y1 - y2) * (y1 - y2);
    double dt2 = double(t1 - t2) * (t1 - t2);
    return std::exp(-dx2/(2.0*sigma_x*sigma_x)
                    -dy2/(2.0*sigma_x*sigma_x)
                    -dt2/(2.0*sigma_t*sigma_t));
}



PYBIND11_MODULE(particle_tracking, m) {
    py::class_<ParticleResult>(m, "ParticleResult")
        .def_readonly("particle_id", &ParticleResult::particle_id)
        .def_readonly("centroid_history", &ParticleResult::centroid_history)
        .def_readonly("events", &ParticleResult::events);

    m.def("track_particles_cpp",
          [](const std::vector<std::tuple<int,int,int,float>>& data,
             double sigma_x, double sigma_t,
             double gaussian_threshold,
             int m_threshold)
          {
              if (data.empty())
                  throw std::invalid_argument("Input data is empty.");

              std::vector<Particle> active, finished;
              int id_counter = 0;

              for (auto const& evt : data) {
                  int x, y, pol;
                  float t;
                  std::tie(x, y, pol, t) = evt;

                  std::vector<size_t> overlap;
                  for (size_t i = 0; i < active.size(); ++i) {
                      for (auto const& re : active[i].recent_events) {
                          double score = gaussian_distance(
                              x, y, t,
                              std::get<0>(re), std::get<1>(re), std::get<3>(re),
                              sigma_x, sigma_t);
                          if (score >= gaussian_threshold) {
                              overlap.push_back(i);
                              break;
                          }
                      }
                  }

                  if (overlap.empty()) {
                      active.emplace_back(++id_counter, x, y, pol, t);
                  } else {
                      size_t base = overlap[0];
                      active[base].add_event(x, y, pol, t);
                      if (overlap.size() > 1) {
                          std::vector<size_t> to_merge(overlap.begin()+1, overlap.end());
                          std::sort(to_merge.begin(), to_merge.end(), std::greater<size_t>());
                          for (auto idx : to_merge) {
                              active[base].merge(active[idx]);
                              active.erase(active.begin() + idx);
                          }
                      }
                  }

                  // Prune inactive tracks
                  for (auto it = active.begin(); it != active.end();) {
                      if (!it->is_active(t, m_threshold)) {
                          finished.push_back(*it);
                          it = active.erase(it);
                      } else {
                          ++it;
                      }
                  }
              }

              // Finalize remaining tracks
              for (auto const& p : active) finished.push_back(p);
              finished.erase(
                  std::remove_if(finished.begin(), finished.end(),
                                 [&](Particle const& p) { return !p.is_active_final(m_threshold); }),
                  finished.end());

              std::vector<ParticleResult> results;
              results.reserve(finished.size());
              for (auto const& p : finished) results.push_back(p.get_result());
              return results;
          },
          py::arg("data"),
          py::arg("sigma_x"),
          py::arg("sigma_t"),
          py::arg("gaussian_threshold"),
          py::arg("m_threshold"),
          "Track particles using Gaussian threshold with fixed merge weights and original pruning");
}