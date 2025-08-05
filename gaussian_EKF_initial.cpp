// gating and pruning parameters
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
// gating and pruning parameters
static constexpr double GATE_THRESH = 1.5;       // moderate spatial gating
static constexpr float PRUNE_WINDOW_US = 10000.0f; // prune if no events in last 8 ms
static constexpr float GATE_TIME_US = 10000.0f;   // allow up to 10 ms between events
static constexpr double MAX_ASSOC_DIST = 20.0;    // max pixels for association
static constexpr int MIN_EVENTS_TO_DRAW = 10;   // allow drawing after 2 events
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <deque>
#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <random>

using Event = std::tuple<int,int,float>; // x,y,time(us)
using Centroid = std::tuple<float,double,double>;

struct EKFTrack {
    int id;
    Eigen::Matrix<double,7,1> x;  // [px, py, vx, vy, l1, l2, theta]
    Eigen::Matrix<double,7,7> P;  // covariance
    std::deque<Event> recent;    // store recent events for variance update
    Eigen::Matrix2d LambdaInv;  // cached inverse of shape covariance

    EKFTrack(int _id, int x0, int y0, float t0) : id(_id) {
        // initialize state: position at (x0,y0), zero velocity, default shape
        x << x0, y0, 0.0, 0.0, 10.0, 10.0, 0.0;   // moderate initial blob size
        P = Eigen::Matrix<double,7,7>::Identity() * 100.0;
        // add initial event for pruning and chi2
        recent.emplace_back(x0, y0, t0);
        // initialize LambdaInv based on default shape
        double l1 = x(4), l2 = x(5), th = x(6);
        double inv_l1 = 1.0 / l1, inv_l2 = 1.0 / l2;
        Eigen::Matrix2d R;
        R << std::cos(th), -std::sin(th),
             std::sin(th),  std::cos(th);
        Eigen::Matrix2d Dinv;
        Dinv << inv_l1, 0,
                0,     inv_l2;
        LambdaInv = R * Dinv * R.transpose();
    }

    // add a new event and prune older than 2ms window
    void addEvent(int ex, int ey, float et) {
        recent.emplace_back(ex, ey, et);
        float cutoff = et - 2000.0f;
        while (!recent.empty() && std::get<2>(recent.front()) < cutoff)
            recent.pop_front();
    }

    void predict(double dt) {
        // state transition matrix
        Eigen::Matrix<double,7,7> F = Eigen::Matrix<double,7,7>::Identity();
        F(0,2) = dt;
        F(1,3) = dt;
        // process noise
        Eigen::Matrix<double,7,7> Q = Eigen::Matrix<double,7,7>::Identity() * 1e-3;
        x = F * x;
        P = F * P * F.transpose() + Q;

        // cache the Mahalanobis inverse for gating
        double l1 = x(4), l2 = x(5), th = x(6);
        double inv_l1 = 1.0 / l1;
        double inv_l2 = 1.0 / l2;
        Eigen::Matrix2d R;
        R << std::cos(th), -std::sin(th),
             std::sin(th),  std::cos(th);
        Eigen::Matrix2d Dinv;
        Dinv << inv_l1, 0,
                0,     inv_l2;
        LambdaInv = R * Dinv * R.transpose();
    }

    void updatePosition(double zx, double zy) {
        Eigen::Vector2d z(zx, zy);
        Eigen::Vector2d h = x.template segment<2>(0);
        Eigen::Matrix<double,2,7> H = Eigen::Matrix<double,2,7>::Zero();
        H(0,0) = 1; H(1,1) = 1;
        Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * 0.5;
        Eigen::Matrix2d S = H * P * H.transpose() + R;
        Eigen::Matrix<double,7,2> K = P * H.transpose() * S.inverse();
        x += K * (z - h);
        P = (Eigen::Matrix<double,7,7>::Identity() - K * H) * P;
    }

    void updateVariance(double chi2) {
        // simple pseudo-measurement: total variance = lambda1 + lambda2
        double vz = chi2;
        double h = x(4) + x(5);
        Eigen::Matrix<double,1,7> H = Eigen::Matrix<double,1,7>::Zero();
        H(0,4) = 1; H(0,5) = 1;
        double R = 4.0;
        double S = H * P * H.transpose() + R;
        Eigen::Matrix<double,7,1> K = P * H.transpose() / S;
        x += K * (vz - h);
        P = (Eigen::Matrix<double,7,7>::Identity() - K * H) * P;
    }
};
// Mahalanobis gating wrapper using current EKFTrack shape with temporal gating
static inline bool mahalanobis_gate(const EKFTrack &trk, int x, int y, float t) {
    // temporal gating: ensure event is recent relative to track
    if (trk.recent.empty() || (t - std::get<2>(trk.recent.back()) > GATE_TIME_US))
        return false;
    // absolute pixel distance gating
    double dx0 = double(x) - trk.x(0);
    double dy0 = double(y) - trk.x(1);
    if (dx0*dx0 + dy0*dy0 > MAX_ASSOC_DIST*MAX_ASSOC_DIST)
        return false;
    Eigen::Vector2d d(double(x - trk.x(0)), double(y - trk.x(1)));
    return (d.transpose() * trk.LambdaInv * d <= GATE_THRESH);
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " data.csv\n";
        return 1;
    }
    std::ifstream ifs(argv[1]);
    if(!ifs) throw std::runtime_error("Cannot open file");
    std::vector<Event> events;
    std::string line;
    while(std::getline(ifs, line)) {
        std::stringstream ss(line);
        int x, y, p; float t; char c;
        if(!(ss>>x>>c>>y>>c>>p>>c>>t)) continue;
        if(p!=1) continue;
        events.emplace_back(x,y,t);
    }
    if(events.empty()) return 0;
    std::sort(events.begin(), events.end(), [](auto &a, auto &b){ return std::get<2>(a) < std::get<2>(b); });

    // Parameters
    float bg_window_us = 50000.0f;

    std::vector<EKFTrack> active;
    int next_id = 0;
    std::map<int, cv::Scalar> color_map;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(50,255);
    std::deque<Event> bg_events;

    // Fixed frame rate
    const int fps = 30;
    const int frame_loop_ms = 1000 / fps;
    size_t idx = 0;
    float start_time = std::get<2>(events[0]);

    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::Mat frame(720,1280,CV_8UC3);

    // Main loop: fixed-rate frames
    auto wall_start = std::chrono::high_resolution_clock::now();
    while(true) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(now - wall_start).count();
        float speed_factor = 0.1f;  // 再生速度倍率
        float play_time = start_time + elapsed_ms * 1000.0f * speed_factor;

        // consume events
        while(idx < events.size() && std::get<2>(events[idx]) <= play_time) {
            auto [x,y,t] = events[idx++];
            bg_events.emplace_back(x,y,t);
            float cutoff_bg = t - bg_window_us;
            while(!bg_events.empty() && std::get<2>(bg_events.front()) < cutoff_bg)
                bg_events.pop_front();

            static float last_time = start_time;
            double dt = (t - last_time) * 1e-6;  // convert microseconds to seconds
            last_time = t;
            for(auto &trk : active) trk.predict(dt);

            // Association
            std::vector<size_t> overlap;
            for(size_t i=0; i<active.size(); ++i) {
                if (mahalanobis_gate(active[i], x, y, t)) {
                    overlap.push_back(i);
                }
            }
            if(overlap.empty()) {
                active.emplace_back(++next_id, x, y, t);
                color_map[next_id] = cv::Scalar(dist(rng),dist(rng),dist(rng));
            } else {
                size_t base = overlap[0];
                auto &trk = active[base];
                trk.addEvent(x, y, t);
                trk.updatePosition(x, y);
                // compute chi2 from trk.recent
                double chi2 = 0;
                for (auto &e : trk.recent) {
                    double dx = trk.x(0) - std::get<0>(e);
                    double dy = trk.x(1) - std::get<1>(e);
                    chi2 += dx*dx + dy*dy;
                }
                trk.updateVariance(chi2);
                if(overlap.size()>1) {
                    std::sort(overlap.begin()+1, overlap.end(), std::greater<size_t>());
                    for(size_t j=1; j<overlap.size(); ++j) {
                        size_t idx2 = overlap[j];
                        active.erase(active.begin()+idx2);
                    }
                }
            }

            // prune tracks with no recent events in PRUNE_WINDOW_US
            for(auto it = active.begin(); it != active.end();) {
                if (std::get<2>(it->recent.back()) < (play_time - PRUNE_WINDOW_US)) {
                    it = active.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // 描画
        frame.setTo(cv::Scalar(128,128,128));

        // 1) 各トラックの recent イベントを色分けして描画
        for(auto &trk : active) {
            cv::Scalar col = color_map[trk.id];
            for(auto &e : trk.recent) {
                cv::circle(frame, cv::Point(std::get<0>(e),std::get<1>(e)), 1, col, cv::FILLED);
            }
        }

        // 2) センチロイド（追跡結果）の描画
        for(auto &trk : active) {
            // skip tracks with too few events to be valid
            if (int(trk.recent.size()) < MIN_EVENTS_TO_DRAW) continue;
            int cx = int(trk.x(0)), cy = int(trk.x(1));
            int radius = 3 + std::min(int(trk.recent.size()), 10);
            cv::Scalar col = color_map[trk.id];
            cv::circle(frame, cv::Point(cx,cy), radius, col, cv::FILLED);
            cv::circle(frame, cv::Point(cx,cy), radius, cv::Scalar(0,0,0), 2);
        }

        cv::imshow("Tracking", frame);
        if(cv::waitKey(frame_loop_ms) == 27) break;
        if(idx >= events.size()) break;
    }
    return 0;
}