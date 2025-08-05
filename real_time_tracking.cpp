// standard library
#include <deque>
#include <tuple>
#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>

// third-party libraries
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>

// gating and pruning parameters (units: microseconds/us)
static constexpr float PRUNE_WINDOW_US      = 10000.0f;  // prune if no events in last 10 ms
static constexpr float GATE_TIME_US         = 10000.0f;  // allow up to 10 ms between events
static constexpr double MAX_ASSOC_DIST      =    20.0;   // max pixels for association
static constexpr double GATE_THRESH         =     1.6;   // Mahalanobis gating threshold
static constexpr int   MIN_EVENTS_TO_DRAW   =       10;   // draw only if ≥10 recent events
static constexpr size_t RECENT_BUFFER_SIZE  =    500;
static constexpr size_t BG_BUFFER_SIZE      =    500;

using Event    = std::tuple<int,int,double>;  // x, y, timestamp(us)
using Centroid = std::tuple<double,double,double>;

struct EKFTrack {
    int id;
    Eigen::Matrix<double,7,1> x;   // [px, py, vx, vy, l1, l2, theta]
    Eigen::Matrix<double,7,7> P;   // covariance
    boost::circular_buffer<Event> recent;
    Eigen::Matrix2d LambdaInv;

    EKFTrack(int _id, int x0, int y0, double t0)
      : id(_id), recent(RECENT_BUFFER_SIZE)
    {
        x << x0, y0, 0, 0, 10, 10, 0;
        P = Eigen::Matrix<double,7,7>::Identity() * 100.0;
        recent.push_back(Event{x0,y0,t0});
        updateLambdaInv();
    }

    void addEvent(int ex, int ey, double et) {
        recent.push_back(Event{ex,ey,et});
        while (!recent.empty() && (et - std::get<2>(recent.front())) > PRUNE_WINDOW_US)
            recent.pop_front();
    }

    void predict(double dt) {
        Eigen::Matrix<double,7,7> F = Eigen::Matrix<double,7,7>::Identity();
        F(0,2)=dt; F(1,3)=dt;
        Eigen::Matrix<double,7,7> Q = Eigen::Matrix<double,7,7>::Identity() * 1e-3;
        x = F*x;
        P = F*P*F.transpose() + Q;
        updateLambdaInv();
    }

    void updatePosition(double zx, double zy) {
        Eigen::Vector2d z(zx, zy);
        Eigen::Vector2d h = x.template segment<2>(0);
        Eigen::Matrix<double,2,7> H = Eigen::Matrix<double,2,7>::Zero();
        H(0,0)=1; H(1,1)=1;
        Eigen::Matrix2d R = Eigen::Matrix2d::Identity()*0.5;
        Eigen::Matrix2d S = H*P*H.transpose() + R;
        auto K = P*H.transpose()*S.inverse();
        x += K*(z - h);
        P = (Eigen::Matrix<double,7,7>::Identity() - K*H)*P;
    }

    void updateVariance(double chi2) {
        double vz = chi2;
        double h  = x(4)+x(5);
        Eigen::Matrix<double,1,7> H = Eigen::Matrix<double,1,7>::Zero();
        H(0,4)=1; H(0,5)=1;
        double R = 4.0;
        double S = (H*P*H.transpose())(0,0) + R;
        auto K = P*H.transpose()/S;
        x += K*(vz - h);
        P = (Eigen::Matrix<double,7,7>::Identity() - K*H)*P;
    }

private:
    void updateLambdaInv() {
        double l1 = x(4), l2 = x(5), th = x(6);
        double i1 = 1.0/l1, i2 = 1.0/l2;
        Eigen::Matrix2d R;
        R << std::cos(th), -std::sin(th),
             std::sin(th),  std::cos(th);
        Eigen::Matrix2d Dinv;
        Dinv << i1, 0,
                0,  i2;
        LambdaInv = R * Dinv * R.transpose();
    }
};

static inline bool mahalanobis_gate(const EKFTrack &trk, int x, int y, double t) {
    if (trk.recent.empty() || (t - std::get<2>(trk.recent.back())) > GATE_TIME_US)
        return false;
    double dx = double(x) - trk.x(0);
    double dy = double(y) - trk.x(1);
    if (dx*dx + dy*dy > MAX_ASSOC_DIST*MAX_ASSOC_DIST)
        return false;
    Eigen::Vector2d d(dx, dy);
    return (d.transpose() * trk.LambdaInv * d <= GATE_THRESH);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " data.csv\n";
        return 1;
    }

    // --- 1) Load events from CSV ---
    std::ifstream ifs(argv[1]);
    if (!ifs) throw std::runtime_error("Cannot open file");
    std::vector<Event> events;
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        int x, y, p; double t; char c;
        if (!(ss>>x>>c>>y>>c>>p>>c>>t)) continue;
        if (p != 1) continue;
        events.emplace_back(x,y,t);
    }
    if (events.empty()) return 0;
    std::sort(events.begin(), events.end(),
              [](auto &a, auto &b){ return std::get<2>(a) < std::get<2>(b); });

    // --- 2) Tracking & drawing setup ---
    std::vector<EKFTrack> active;
    int next_id = 0;
    std::map<int,cv::Scalar> color_map;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> color_dist(50,255);

    const double sampling_ratio = 0.1;               // sample 10% of events for display
    std::uniform_real_distribution<double> uni(0,1);

    const int display_fps       = 30;                // draw at 10 FPS
    auto display_interval       = std::chrono::milliseconds(1000/display_fps);
    auto next_draw_time         = std::chrono::steady_clock::now() + display_interval;

    std::vector<cv::Point> disp_events;
    cv::Mat frame(720,1280,CV_8UC3);

    // --- 3) Main loop ---
    double last_time = std::get<2>(events.front());
    double start_time = last_time;
    static int processed = 0;
    for (auto &ev : events) {
        int ex = std::get<0>(ev), ey = std::get<1>(ev);
        double t = std::get<2>(ev);

        // a) Predict existing tracks
        double dt = (t - last_time)*1e-6;
        last_time = t;
        for (auto &trk : active) trk.predict(dt);

        // b) Associate / create tracks
        std::vector<size_t> hits;
        for (size_t i=0; i<active.size(); ++i) {
            if (mahalanobis_gate(active[i], ex, ey, t))
                hits.push_back(i);
        }
        if (hits.empty()) {
            active.emplace_back(++next_id, ex, ey, t);
            color_map[next_id] = cv::Scalar(
                color_dist(rng), color_dist(rng), color_dist(rng)
            );
        } else {
            auto &trk = active[hits[0]];
            trk.addEvent(ex, ey, t);
            trk.updatePosition(ex, ey);
            // compute chi2
            double chi2 = 0;
            for (auto &e : trk.recent) {
                double dx = trk.x(0) - std::get<0>(e);
                double dy = trk.x(1) - std::get<1>(e);
                chi2 += dx*dx + dy*dy;
            }
            trk.updateVariance(chi2);
        }

        // c) Prune old tracks
        active.erase(
            std::remove_if(active.begin(), active.end(),
                [&](const EKFTrack &trk){
                    return (t - std::get<2>(trk.recent.back())) > PRUNE_WINDOW_US;
                }),
            active.end()
        );

        // d) Sampling for display
        if (uni(rng) < sampling_ratio) {
            disp_events.emplace_back(ex, ey);
        }

        // count events for this frame
        processed++;

        // e) Time to draw?
        auto now = std::chrono::steady_clock::now();
        if (now >= next_draw_time) {
            // clear frame
            frame.setTo(cv::Scalar(30,30,30));

            // draw sampled events
            for (auto &pt : disp_events) {
                cv::rectangle(frame,
                              pt - cv::Point(1,1),
                              pt + cv::Point(1,1),
                              cv::Scalar(200,200,200),
                              cv::FILLED);
            }

            // draw track centroids
            for (auto &trk : active) {
                if (int(trk.recent.size()) < MIN_EVENTS_TO_DRAW) continue;
                cv::Point cen(int(trk.x(0)), int(trk.x(1)));
                cv::circle(frame, cen, 4, color_map[trk.id], 10,cv::FILLED);
            }

            // draw timestamp
            char buf[64];
            double elapsed_s = (t - start_time) * 1e-6;
            std::snprintf(buf, sizeof(buf), "t=%.3f s", elapsed_s);
            cv::putText(frame, buf, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);

            // draw processed event count
            char buf3[64];
            std::snprintf(buf3, sizeof(buf3), "evts=%d", processed);
            cv::putText(frame, buf3, cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 1);

            // show
            cv::imshow("Tracking", frame);
            if (cv::waitKey(1) == 27) break;

            // reset for next frame
            disp_events.clear();
            processed = 0;
            next_draw_time += display_interval;
        }
    }

    cv::waitKey(0);
    return 0;
}
