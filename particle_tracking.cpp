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
struct ParticleResult {
    int particle_id;
    std::vector<std::tuple<float, double, double>> centroid_history;
    std::vector<std::tuple<int, int, int, float>> events;
};

// Particle class
class Particle {
public:
    int particle_id;
    std::vector<std::tuple<int,int,int,float>> events;
    std::deque<std::tuple<int,int,int,float>> recent_events;
    std::vector<std::tuple<float,double,double>> centroid_history;
    int mass;
    float window_size_us;
    float decay_tau;

    Particle(int id, int x, int y, int polarity, float time, float window_us)
        : particle_id(id), mass(1), window_size_us(window_us), decay_tau(window_us) {
        events.emplace_back(x, y, polarity, time);
        recent_events.emplace_back(x, y, polarity, time);
        centroid_history.emplace_back(time, x, y);
    }

    void update_centroid(float time) {
        double sum_x = 0.0, sum_y = 0.0;
        for (const auto& evt : recent_events) {
            sum_x += std::get<0>(evt);
            sum_y += std::get<1>(evt);
        }
        double centroid_x = sum_x / recent_events.size();
        double centroid_y = sum_y / recent_events.size();
        centroid_history.emplace_back(time, centroid_x, centroid_y);
    }

    void add_event(int x, int y, int polarity, float time) {
        events.emplace_back(x, y, polarity, time);
        recent_events.emplace_back(x, y, polarity, time);
        mass++;
        float cutoff_time = time - window_size_us;
        while (!recent_events.empty() && std::get<3>(recent_events.front()) < cutoff_time) {
            recent_events.pop_front();
        }
        update_centroid(time);
    }

    void merge(const Particle& other) {
        for (const auto& evt : other.events) {
            events.push_back(evt);
            recent_events.push_back(evt);
        }
        int original_mass = mass;
        mass += other.mass;
        double this_x = std::get<1>(centroid_history.back());
        double this_y = std::get<2>(centroid_history.back());
        double other_x = std::get<1>(other.centroid_history.back());
        double other_y = std::get<2>(other.centroid_history.back());
        double total_mass = original_mass + other.mass;
        double merged_x = (this_x * original_mass + other_x * other.mass) / total_mass;
        double merged_y = (this_y * original_mass + other_y * other.mass) / total_mass;
        float latest_time = std::get<3>(recent_events.back());
        centroid_history.emplace_back(latest_time, merged_x, merged_y);
    }

    float current_confidence(float current_time) const {
        float dt = current_time - std::get<3>(events.back());
        return std::exp(-dt / decay_tau);
    }

    bool is_active(float current_time, int m_threshold) const {
        if (current_confidence(current_time) < 0.5f) return false;
        if (std::get<3>(events.back()) < current_time - window_size_us)
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

// Elliptical top-hat overlap: normalized squared distance <= 1
static bool elliptical_tophat(int x1, int y1, float t1,
                              int x2, int y2, float t2,
                              double sigma_x, double sigma_t) {
    double dx = (x1 - x2) / sigma_x;
    double dy = (y1 - y2) / sigma_x;
    double dt = (t1 - t2) / sigma_t;
    return dx*dx + dy*dy + dt*dt <= 1.0;
}

PYBIND11_MODULE(particle_tracking, m) {
    py::class_<ParticleResult>(m, "ParticleResult")
        .def_readonly("particle_id", &ParticleResult::particle_id)
        .def_readonly("centroid_history", &ParticleResult::centroid_history)
        .def_readonly("events", &ParticleResult::events);

    m.def("track_particles_cpp", [](const std::vector<std::tuple<int,int,int,float>>& data,
                                     double sigma_x, double sigma_t,
                                     int m_threshold, float window_size_us) {
        if (data.empty()) throw std::invalid_argument("Input data is empty.");
        std::vector<Particle> active, finished;
        int id_counter = 0;

        for (auto const& evt : data) {
            int x, y, pol; float t;
            std::tie(x, y, pol, t) = evt;
            std::vector<size_t> overlap;
            for (size_t i = 0; i < active.size(); ++i) {
                for (auto const& re : active[i].recent_events) {
                    if (elliptical_tophat(x, y, t,
                                          std::get<0>(re), std::get<1>(re), std::get<3>(re),
                                          sigma_x, sigma_t)) {
                        overlap.push_back(i);
                        break;
                    }
                }
            }
            if (overlap.empty()) {
                active.emplace_back(++id_counter, x, y, pol, t, window_size_us);
            } else {
                size_t base = overlap[0];
                active[base].add_event(x, y, pol, t);
                if (overlap.size() > 1) {
                    std::vector<size_t> to_merge(overlap.begin()+1, overlap.end());
                    std::sort(to_merge.begin(), to_merge.end(), std::greater<size_t>());
                    for (size_t idx : to_merge) active[base].merge(active[idx]);
                    for (size_t idx : to_merge) active.erase(active.begin() + idx);
                }
            }
            for (auto it = active.begin(); it != active.end(); ) {
                if (!it->is_active(t, m_threshold)) {
                    finished.push_back(*it);
                    it = active.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto const& p : active) finished.push_back(p);
        finished.erase(std::remove_if(finished.begin(), finished.end(),
            [&](Particle const& p){ return !p.is_active_final(m_threshold); }), finished.end());
        std::vector<ParticleResult> results;
        results.reserve(finished.size());
        for (auto const& p : finished) results.push_back(p.get_result());
        return results;
    }, py::arg("data"), py::arg("sigma_x"), py::arg("sigma_t"),
       py::arg("m_threshold"), py::arg("window_size_us")=2000.0f);
}
