#include <opencv2/opencv.hpp>
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

    std::vector<Particle> active;
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

            // Association
            std::vector<size_t> overlap;
            for(size_t i=0; i<active.size(); ++i) {
                for(auto &re : active[i].recent) {
                    if(tophat_fast(x,y,t,
                                  std::get<0>(re),std::get<1>(re),std::get<2>(re),
                                  inv_sx, inv_st)) {
                        overlap.push_back(i);
                        break;
                    }
                }
            }
            if(overlap.empty()) {
                active.emplace_back(++next_id, x,y,t);
                color_map[next_id] = cv::Scalar(dist(rng),dist(rng),dist(rng));
            } else {
                size_t base = overlap[0];
                active[base].add(x,y,t);
                if(overlap.size()>1) {
                    std::sort(overlap.begin()+1, overlap.end(), std::greater<size_t>());
                    for(size_t j=1; j<overlap.size(); ++j) {
                        size_t idx2 = overlap[j];
                        active[base].merge(active[idx2]);
                        active.erase(active.begin()+idx2);
                    }
                }
            }

            // remove lost tracks
            for(auto it = active.begin(); it != active.end();) {
                if(std::get<2>(it->recent.back()) < t - 2000.0f) {
                    it = active.erase(it);
                } else ++it;
            }
        }

        // 描画
        frame.setTo(cv::Scalar(128,128,128));

        // 1) 各粒子の recent イベントを粒子色でプロット
        for(auto &p : active) {
            cv::Scalar col = color_map[p.id];
            for(auto &e : p.recent) {
                int ex = std::get<0>(e);
                int ey = std::get<1>(e);
                cv::circle(frame, cv::Point(ex,ey), 1, col, cv::FILLED);
            }
        }

        // 2) センチロイド（追跡結果）の描画
        for(auto &p : active) {
            if(p.mass <= 10) continue;
            auto &h = p.history.back();
            int cx = int(std::get<1>(h));
            int cy = int(std::get<2>(h));
            int radius = 3 + std::min(p.mass,5);
            cv::Scalar col = color_map[p.id];
            cv::circle(frame, cv::Point(cx,cy), radius, col, cv::FILLED);
            cv::circle(frame, cv::Point(cx,cy), radius, cv::Scalar(0,0,0), 3);
        }

        cv::imshow("Tracking", frame);
        if(cv::waitKey(frame_loop_ms) == 27) break;
        if(idx >= events.size()) break;
    }
    return 0;
}