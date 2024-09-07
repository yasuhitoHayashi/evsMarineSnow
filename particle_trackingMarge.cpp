#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <deque>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <limits>  // for std::numeric_limits

// Struct to hold the result for each particle, including centroid history
struct ParticleResult {
    int particle_id;
    std::vector<std::tuple<float, double, double>> centroid_history;  // (time, centroid_x, centroid_y)
    std::vector<std::tuple<int, int, float>> events;  // List of (x, y, time) events with float time
};

class Particle {
public:
    int particle_id;
    std::deque<std::tuple<int, int, float>> events;  // (x, y, time) events with float time
    std::deque<std::tuple<int, int, float>> recent_events;  // Recent events for centroid calculation
    double centroid_x, centroid_y;
    int mass;
    std::deque<std::tuple<float, double, double>> centroid_history;  // (time, centroid_x, centroid_y)

    Particle(int id, int x, int y, float time) : particle_id(id), centroid_x(x), centroid_y(y), mass(1) {
        events.push_back(std::make_tuple(x, y, time));
        recent_events.push_back(std::make_tuple(x, y, time));  // Add to recent events as well
        centroid_history.push_back(std::make_tuple(time, centroid_x, centroid_y));
    }

    // マージ用の関数
    void merge(const Particle& other) {
        // 他の粒子のイベントを全て追加
        for (const auto& event : other.events) {
            events.push_back(event);
            recent_events.push_back(event);  // recent_eventsにも追加
        }

        // 質量の統合
        mass += other.mass;

        // 重心の再計算（質量で重み付けした座標平均）
        double total_mass = mass + other.mass;
        centroid_x = (centroid_x * mass + other.centroid_x * other.mass) / total_mass;
        centroid_y = (centroid_y * mass + other.centroid_y * other.mass) / total_mass;

        // 重心の履歴を更新
        if (!recent_events.empty()) {
            float time = std::get<2>(recent_events.back());
            centroid_history.push_back(std::make_tuple(time, centroid_x, centroid_y));
        }
    }

    void add_event(int x, int y, float time) {
        events.push_back(std::make_tuple(x, y, time));
        recent_events.push_back(std::make_tuple(x, y, time));  // Add to recent events
        mass++;

        // Remove old events from recent_events (older than 2000us)
        float cutoff_time = time - 2000.0;
        while (!recent_events.empty() && std::get<2>(recent_events.front()) < cutoff_time) {
            recent_events.pop_front();
        }

        // Calculate centroid based on recent events
        if (!recent_events.empty()) {
            double sum_x = 0, sum_y = 0;
            for (const auto& event : recent_events) {
                sum_x += std::get<0>(event);
                sum_y += std::get<1>(event);
            }
            centroid_x = sum_x / recent_events.size();
            centroid_y = sum_y / recent_events.size();
        }

        // Save the current centroid to history
        centroid_history.push_back(std::make_tuple(time, centroid_x, centroid_y));
    }

    bool is_active(float current_time, int m_threshold) const {
        // Check if the particle is inactive based on time and mass
        if (!events.empty() && std::get<2>(events.back()) < current_time - 2000.0) {
            return mass > m_threshold;  // Inactive if mass is below the threshold
        }
        return true;  // Otherwise active
    }

    // Final check based on mass only, ignoring time
    bool is_active_final(int m_threshold) const {
        return mass > m_threshold;  // Only mass is considered for the final check
    }

    // Function to return particle results for Python
    ParticleResult get_result() const {
        ParticleResult result;
        result.particle_id = particle_id;
        result.centroid_history = std::vector<std::tuple<float, double, double>>(centroid_history.begin(), centroid_history.end());
        result.events = std::vector<std::tuple<int, int, float>>(events.begin(), events.end());
        return result;
    }
};

// ガウス分布による距離計算
double gaussian_distance(int x1, int y1, float t1, int x2, int y2, float t2, double sigma_x, double sigma_t) {
    double spatial_distance_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  // 空間的な距離の2乗
    double time_distance_sq = (t1 - t2) * (t1 - t2);  // 時間的な距離の2乗
    return std::exp(-spatial_distance_sq / (2 * sigma_x * sigma_x) - time_distance_sq / (2 * sigma_t * sigma_t));  // ガウス分布に基づくスコア
}

std::vector<ParticleResult> track_particles_cpp(const std::vector<std::tuple<int, int, float>>& data, double sigma_x, double sigma_t, double gaussian_threshold, int m_threshold) {
    std::vector<Particle> particles;
    int particle_id_counter = 0;

    // Iterate through each event in the data
    for (const auto& event : data) {
        int x = std::get<0>(event);
        int y = std::get<1>(event);
        float time = std::get<2>(event);

        bool found_overlap = false;
        size_t overlapping_particle_index = std::numeric_limits<size_t>::max();  // 修正: size_tに変更

        // Iterate over each particle to check if the event belongs to one
        for (size_t i = 0; i < particles.size(); ++i) {  // 修正: size_tに変更
            Particle& particle = particles[i];

            // Check all recent events within the last 2000us
            for (const auto& recent_event : particle.recent_events) {
                // ガウス分布に基づいた距離を計算
                double gaussian_score = gaussian_distance(x, y, time, std::get<0>(recent_event), std::get<1>(recent_event), std::get<2>(recent_event), sigma_x, sigma_t);

                // スコアが閾値を超えた場合にイベントを追加
                if (gaussian_score >= gaussian_threshold) {
                    particle.add_event(x, y, time);
                    found_overlap = true;
                    overlapping_particle_index = i;
                    break;
                }
            }
            if (found_overlap) {
                break;
            }
        }

        if (!found_overlap) {
            // Create a new particle if no overlap was found
            particle_id_counter++;
            Particle new_particle(particle_id_counter, x, y, time);
            particles.push_back(new_particle);
        } else if (overlapping_particle_index != std::numeric_limits<size_t>::max()) {
            // If another overlapping particle is found, merge them
            for (size_t i = 0; i < particles.size(); ++i) {
                if (i != overlapping_particle_index) {
                    Particle& particle = particles[i];

                    // ガウス分布に基づく距離計算で、同じ条件で2つの粒子が重なっているかチェック
                    for (const auto& recent_event : particle.recent_events) {
                        double gaussian_score = gaussian_distance(
                            std::get<0>(particles[overlapping_particle_index].recent_events.back()),
                            std::get<1>(particles[overlapping_particle_index].recent_events.back()),
                            std::get<2>(particles[overlapping_particle_index].recent_events.back()),
                            std::get<0>(recent_event),
                            std::get<1>(recent_event),
                            std::get<2>(recent_event),
                            sigma_x, sigma_t);

                        if (gaussian_score >= gaussian_threshold) {
                            // Merge the particles
                            particles[overlapping_particle_index].merge(particle);
                            particles.erase(particles.begin() + i);  // Merge後、消す
                            break;
                        }
                    }
                }
            }
        }

        // Remove inactive particles based on time and mass
        particles.erase(std::remove_if(particles.begin(), particles.end(),
            [time, m_threshold](const Particle& p) { return !p.is_active(time, m_threshold); }), particles.end());
    }

    // At the end, check for particles based only on mass
    particles.erase(std::remove_if(particles.begin(), particles.end(),
        [m_threshold](const Particle& p) { return !p.is_active_final(m_threshold); }), particles.end());

    // Return results as a vector of ParticleResult for Python
    std::vector<ParticleResult> results;
    for (const auto& particle : particles) {
        results.push_back(particle.get_result());
    }

    return results;
}

PYBIND11_MODULE(particle_tracking, m) {
    pybind11::class_<ParticleResult>(m, "ParticleResult")
        .def_readonly("particle_id", &ParticleResult::particle_id)
        .def_readonly("centroid_history", &ParticleResult::centroid_history)
        .def_readonly("events", &ParticleResult::events);

    m.def("track_particles_cpp", &track_particles_cpp, "Track particles in C++");
}