#ifndef PARTICLE_TRACKING_HPP
#define PARTICLE_TRACKING_HPP

#include <cstdint>
#include <tuple>
#include <vector>

struct ParticleResult {
  int particle_id;
  std::vector<std::tuple<int, int, std::int64_t>> events; // (x,y,t_us)
};

std::vector<ParticleResult>
track_particles(const std::vector<std::tuple<int, int, std::int64_t>> &data,
                double sigma_x, double sigma_t, int m_threshold,
                std::int64_t recent_window_us, std::int64_t retire_window_us);

#endif // PARTICLE_TRACKING_HPP
