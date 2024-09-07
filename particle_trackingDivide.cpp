#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <deque>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <Eigen/Dense>  // カルマンフィルタに必要なEigenライブラリ

// Struct to hold the result for each particle, including centroid history
struct ParticleResult {
    int particle_id;
    std::vector<std::tuple<float, double, double>> centroid_history;  // (time, centroid_x, centroid_y)
    std::vector<std::tuple<int, int, float>> events;  // List of (x, y, time) events with float time
};

class KalmanFilter {
public:
    Eigen::Vector4d state;  // 状態ベクトル [x, y, vx, vy]
    Eigen::Matrix4d P;      // 共分散行列
    Eigen::Matrix4d F;      // 状態遷移行列
    Eigen::Matrix2d R;      // 観測ノイズ共分散行列
    Eigen::Matrix<double, 2, 4> H;  // 観測行列

    KalmanFilter(double x, double y) {
        // 初期状態
        state << x, y, 0, 0;

        // 共分散行列の初期化
        P = Eigen::Matrix4d::Identity() * 1000.0;

        // 状態遷移行列の設定 (dt=1のシンプルな例)
        F << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

        // 観測行列の設定
        H << 1, 0, 0, 0,
             0, 1, 0, 0;

        // 観測ノイズ共分散行列の設定
        R = Eigen::Matrix2d::Identity() * 1.0;
    }

    void predict() {
        // 状態の予測
        state = F * state;

        // 共分散行列の予測
        P = F * P * F.transpose();
    }

    void update(double x, double y) {
        Eigen::Vector2d z;
        z << x, y;

        Eigen::Vector2d y_ = z - H * state;
        Eigen::Matrix2d S = H * P * H.transpose() + R;
        Eigen::Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();

        state = state + K * y_;
        P = (Eigen::Matrix4d::Identity() - K * H) * P;
    }

    double getX() const { return state(0); }
    double getY() const { return state(1); }
};

class Particle {
public:
    int particle_id;
    std::deque<std::tuple<int, int, float>> events;  // (x, y, time) events with float time
    std::deque<std::tuple<int, int, float>> recent_events;  // Recent events for centroid calculation
    double centroid_x, centroid_y;
    int mass;
    std::deque<std::tuple<float, double, double>> centroid_history;  // (time, centroid_x, centroid_y)
    std::vector<int> merge_history;  // 過去のマージ履歴 (他の粒子IDを格納)
    KalmanFilter* kalman_filter = nullptr;  // 粒子の位置を推定するためのカルマンフィルタ（分離時に使用）
    double variance_threshold = 100.0;  // 分離のための分散しきい値
    int mass_threshold_for_split = 10;  // マージや分離を考慮するための質量しきい値

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

        // マージ履歴に追加
        merge_history.push_back(other.particle_id);
        merge_history.insert(merge_history.end(), other.merge_history.begin(), other.merge_history.end());

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

        // 質量が一定以上で分散が大きい場合は分離を考慮
        if (mass > mass_threshold_for_split && calculate_variance() > variance_threshold) {
            split_particle();
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

        // カルマンフィルタは分離が必要な場合にのみ初期化・使用
        if (kalman_filter) {
            kalman_filter->update(centroid_x, centroid_y);
        }

        // Save the current centroid to history
        centroid_history.push_back(std::make_tuple(time, centroid_x, centroid_y));
    }

    double calculate_variance() const {
        // Calculate the spatial variance of the recent events
        double mean_x = 0, mean_y = 0;
        double var_x = 0, var_y = 0;

        for (const auto& event : recent_events) {
            mean_x += std::get<0>(event);
            mean_y += std::get<1>(event);
        }
        mean_x /= recent_events.size();
        mean_y /= recent_events.size();

        for (const auto& event : recent_events) {
            var_x += std::pow(std::get<0>(event) - mean_x, 2);
            var_y += std::pow(std::get<1>(event) - mean_y, 2);
        }

        var_x /= recent_events.size();
        var_y /= recent_events.size();

        return var_x + var_y;  // 空間的な分散
    }

    // 粒子を分離するロジック
    void split_particle() {
        if (!merge_history.empty()) {
            // 過去にマージがあった場合、適切に再マージを行う
            remerge_based_on_history();
        } else {
            // マージ履歴がない場合、通常の分離処理
            simple_split();
        }
    }

    // 軌跡をもとに再マージを行う (カルマンフィルタを使用)
    void remerge_based_on_history() {
        // カルマンフィルタの初期化をここで行う
        if (!kalman_filter) {
            kalman_filter = new KalmanFilter(centroid_x, centroid_y);
        }

        // 過去の粒子の動きをカルマンフィルタで再現し、現在のイベントと照合して関連付け
        std::vector<Particle> new_particles;

        for (const auto& event : recent_events) {
            double predicted_x = kalman_filter->getX();
            double predicted_y = kalman_filter->getY();

            // 予測位置とイベントの位置を比較して関連付け
            double distance = std::sqrt(std::pow(predicted_x - std::get<0>(event), 2) + std::pow(predicted_y - std::get<1>(event), 2));

            if (distance < variance_threshold) {
                // 予測位置に近いイベントを現在の粒子に追加
                add_event(std::get<0>(event), std::get<1>(event), std::get<2>(event));
            } else {
                // 予測位置から離れているイベントは新しい粒子として分離
                Particle new_particle(particle_id, std::get<0>(event), std::get<1>(event), std::get<2>(event));
                new_particles.push_back(new_particle);
            }
        }
    }

    // 単純な分離処理
    void simple_split() {
        // イベントを距離や分散に基づいて新しい粒子に分割
        std::vector<Particle> new_particles;
        Particle* current_particle = nullptr;
        
        for (const auto& event : recent_events) {
            if (!current_particle || distance_from_centroid(std::get<0>(event), std::get<1>(event)) > variance_threshold) {
                // 新しい粒子を作成
                new_particles.push_back(Particle(particle_id, std::get<0>(event), std::get<1>(event), std::get<2>(event)));
                current_particle = &new_particles.back();
            } else {
                current_particle->add_event(std::get<0>(event), std::get<1>(event), std::get<2>(event));
            }
        }
    }

    double distance_from_centroid(int x, int y) const {
        return std::sqrt((centroid_x - x) * (centroid_x - x) + (centroid_y - y) * (centroid_y - y));
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
        for (size_t i = 0; i < particles.size(); ++i) {
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