#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <tuple>
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

    EKFTrack(int _id, int x0, int y0, float t0) : id(_id) {
        // initialize state: position at (x0,y0), zero velocity, default shape
        x << x0, y0, 0.0, 0.0, 5.0, 5.0, 0.0;
        P = Eigen::Matrix<double,7,7>::Identity() * 100.0;
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

// Particle structure
struct Particle {
    int id;
    int mass;
    std::deque<Event> recent;
    std::vector<Centroid> history;

    Particle(int _id,int x,int y,float t):id(_id),mass(1) {
        recent.emplace_back(x,y,t);
        history.emplace_back(t,x,y);
    }
    void add(int x,int y,float t) {
        recent.emplace_back(x,y,t);
        mass++;
        float cutoff = t - 2000.0f;
        while(!recent.empty() && std::get<2>(recent.front()) < cutoff)
            recent.pop_front();
        double sx=0, sy=0;
        for(auto &e : recent) { sx += std::get<0>(e); sy += std::get<1>(e); }
        history.emplace_back(t, sx/recent.size(), sy/recent.size());
    }
    void merge(const Particle &o) {
        int old = mass;
        mass += o.mass;
        auto [t0, x0, y0] = history.back();
        auto [t1, x1, y1] = o.history.back();
        double total = double(old) + double(o.mass);
        double mx = (x0*old + x1*o.mass)/total;
        double my = (y0*old + y1*o.mass)/total;
        history.emplace_back(std::get<2>(recent.back()), mx, my);
        for(auto &e : o.recent) recent.push_back(e);
    }
};

static inline bool tophat_fast(int x1,int y1,float t1,int x2,int y2,float t2,double inv_sx,double inv_st) {
    double dt=(t1-t2)*inv_st;
    double dt2=dt*dt;
    if(dt2>1.0) return false;
    double dx=(x1-x2)*inv_sx;
    double dy=(y1-y2)*inv_sx;
    return dx*dx + dy*dy + dt2 <= 1.0;
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
    double sigma_x = 6.0 * 0.66832;
    double sigma_t = 10000.0 * 0.66832;
    double inv_sx = 1.0 / sigma_x;
    double inv_st = 1.0 / sigma_t;
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
        float speed_factor = 1.0f;  // 再生速度倍率
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
                // We don't have recent events in EKFTrack, so approximate association by distance
                double dx = x - active[i].x(0);
                double dy = y - active[i].x(1);
                double dt_event = (t - start_time) * 1e-6;
                if(dx*dx*inv_sx*inv_sx + dy*dy*inv_sx*inv_sx <= 1.0) {
                    overlap.push_back(i);
                }
            }
            if(overlap.empty()) {
                active.emplace_back(++next_id, x,y,t);
                color_map[next_id] = cv::Scalar(dist(rng),dist(rng),dist(rng));
            } else {
                size_t base = overlap[0];
                // We don't have recent events vector in EKFTrack, so skip chi2 calculation from recent events
                active[base].updatePosition(x, y);
                double chi2 = 0;
                // Commented out because EKFTrack does not have recent events
                // for(auto &re : active[base].recent) {
                //    double dx = x - std::get<0>(re);
                //    double dy = y - std::get<1>(re);
                //    chi2 += dx*dx + dy*dy;
                // }
                active[base].updateVariance(chi2);
                if(overlap.size()>1) {
                    std::sort(overlap.begin()+1, overlap.end(), std::greater<size_t>());
                    for(size_t j=1; j<overlap.size(); ++j) {
                        size_t idx2 = overlap[j];
                        // No merge method for EKFTrack, so just erase
                        active.erase(active.begin()+idx2);
                    }
                }
            }

            // remove lost tracks
            for(auto it = active.begin(); it != active.end();) {
                // No recent events in EKFTrack, so use position covariance or other criteria
                if(it->x(0) < 0 || it->x(0) > 1280 || it->x(1) < 0 || it->x(1) > 720) {
                    it = active.erase(it);
                } else ++it;
            }
        }

        // 描画
        frame.setTo(cv::Scalar(128,128,128));

        // 1) 各粒子の recent イベントを粒子色でプロット
        // Removed as EKFTrack does not have recent events

        // 2) センチロイド（追跡結果）の描画
        for(auto &trk : active) {
            // Use covariance trace or similar for mass approximation
            double mass = trk.P.trace();
            if(mass <= 10) continue;
            int cx = int(trk.x(0));
            int cy = int(trk.x(1));
            int radius = 3 + std::min(int(mass),5);
            cv::Scalar col = color_map[trk.id];
            cv::circle(frame, cv::Point(cx,cy), radius, col, cv::FILLED);
            cv::circle(frame, cv::Point(cx,cy), radius, cv::Scalar(0,0,0), 3);
        }

        cv::imshow("Tracking", frame);
        if(cv::waitKey(frame_loop_ms) == 27) break;
        if(idx >= events.size()) break;
    }
    return 0;
}