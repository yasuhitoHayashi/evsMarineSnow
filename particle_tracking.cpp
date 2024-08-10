#include <vector>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <iostream>

// Struct to represent a particle
struct Particle {
    int particle_id;
    std::deque<std::tuple<double, double, double>> events;  // x, y, time
    double mass;
    double centroid[2];
    std::deque<std::tuple<double, double, double>> recent_events;
    std::vector<std::tuple<double, double>> centroid_history;
    int centroid_time_window;

    Particle(int id, double x, double y, double time) : particle_id(id), mass(1), centroid_time_window(1000) {
        events.emplace_back(x, y, time);
        recent_events.emplace_back(x, y, time);
        centroid[0] = x;
        centroid[1] = y;
        centroid_history.emplace_back(time, x);  // Corrected initialization
    }

    void add_event(double x, double y, double time) {
        events.emplace_back(x, y, time);
        recent_events.emplace_back(x, y, time);
        mass += 1;

        while (!recent_events.empty() && std::get<2>(recent_events.front()) < time - centroid_time_window) {
            recent_events.pop_front();
        }

        if (!recent_events.empty()) {
            double sum_x = 0.0;
            double sum_y = 0.0;
            for (const auto& event : recent_events) {
                sum_x += std::get<0>(event);
                sum_y += std::get<1>(event);
            }
            centroid[0] = sum_x / recent_events.size();
            centroid[1] = sum_y / recent_events.size();
            centroid_history.emplace_back(time, centroid[0]);
        }
    }

    bool is_active(double current_time, double m_threshold) {
        return (!events.empty() && std::get<2>(events.back()) >= current_time - 2000) || mass > m_threshold;
    }
};

// Custom hash function for pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

double calculate_distance_sq(double x1, double y1, double x2, double y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// Function to perform particle tracking
extern "C" {
    void track_particles(double* data, int num_events, int spatial_radius, int time_window, double m_threshold,
                         int* particle_counts, double* output_data, double* centroid_data, double* event_data) {
        std::unordered_map<int, Particle> particles;
        int particle_id_counter = 0;
        std::vector<Particle*> active_particles;
        std::unordered_map<std::pair<int, int>, std::vector<Particle*>, pair_hash> grid;  // Using custom hash
        double spatial_radius_squared = spatial_radius * spatial_radius;

        for (int i = 0; i < num_events; ++i) {
            double x = data[i * 3];
            double y = data[i * 3 + 1];
            double time = data[i * 3 + 2];

            int ix = static_cast<int>(x);
            int iy = static_cast<int>(y);

            std::vector<Particle*> nearby_particles;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    auto it = grid.find(std::make_pair(ix + dx, iy + dy));
                    if (it != grid.end()) {
                        for (Particle* p : it->second) {
                            if (calculate_distance_sq(p->centroid[0], p->centroid[1], x, y) <= spatial_radius_squared &&
                                std::get<2>(p->events.back()) >= time - time_window) {
                                nearby_particles.push_back(p);
                            }
                        }
                    }
                }
            }

            Particle* nearby_particle = nullptr;
            if (!nearby_particles.empty()) {
                nearby_particle = nearby_particles.front();
                nearby_particle->add_event(x, y, time);
            } else {
                particle_id_counter += 1;
                Particle new_particle(particle_id_counter, x, y, time);
                // Insert directly into the map and get a reference
                auto result = particles.emplace(particle_id_counter, new_particle);
                Particle* p = &(result.first->second);
                active_particles.push_back(p);
                grid[std::make_pair(ix, iy)].push_back(p);
            }

            active_particles.erase(std::remove_if(active_particles.begin(), active_particles.end(),
                                                  [&grid, time, m_threshold](Particle* p) {
                                                      if (!p->is_active(time, m_threshold)) {
                                                          int px = static_cast<int>(p->centroid[0]);
                                                          int py = static_cast<int>(p->centroid[1]);
                                                          grid[std::make_pair(px, py)].erase(
                                                              std::remove(grid[std::make_pair(px, py)].begin(),
                                                                          grid[std::make_pair(px, py)].end(), p),
                                                              grid[std::make_pair(px, py)].end());
                                                          return true;
                                                      }
                                                      return false;
                                                  }),
                               active_particles.end());
        }

        // Output the particle data
        *particle_counts = particles.size();
        int index = 0;
        int centroid_index = 0;
        int event_index = 0;

        for (const auto& kv : particles) {
            const Particle& p = kv.second;
            output_data[index * 5] = static_cast<double>(kv.first);
            output_data[index * 5 + 1] = p.centroid[0];
            output_data[index * 5 + 2] = p.centroid[1];
            output_data[index * 5 + 3] = static_cast<double>(p.mass);
            output_data[index * 5 + 4] = static_cast<double>(p.events.size());
            ++index;

            // Store centroid history
            for (const auto& history : p.centroid_history) {
                centroid_data[centroid_index * 3] = static_cast<double>(kv.first);
                centroid_data[centroid_index * 3 + 1] = std::get<0>(history);  // time
                centroid_data[centroid_index * 3 + 2] = std::get<1>(history);  // centroid x
                ++centroid_index;
            }

            // Store event coordinates
            for (const auto& event : p.events) {
                event_data[event_index * 4] = static_cast<double>(kv.first);
                event_data[event_index * 4 + 1] = std::get<0>(event);  // x
                event_data[event_index * 4 + 2] = std::get<1>(event);  // y
                event_data[event_index * 4 + 3] = std::get<2>(event);  // time
                ++event_index;
            }
        }
    }
}