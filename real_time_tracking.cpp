// SDL2 for accelerated rendering
#include <SDL2/SDL.h>
#include <limits>
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
#include <iostream>
#include <thread>

// third-party libraries
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>

// gating and pruning parameters (units: microseconds/us)
static constexpr float PRUNE_WINDOW_US      = 5000.0f;  // prune if no events in last 10 ms
static constexpr float GATE_TIME_US         = 5000.0f;  // allow up to 10 ms between events
static constexpr double MAX_ASSOC_DIST      =    50.0;   // max pixels for association
static constexpr double MERGE_DIST = 5.0;  // pixels, distance under which to merge tracks
static constexpr double MERGE_VEL_DIFF = 0.5;  // velocity difference threshold
static constexpr double GATE_THRESH         =     60;   // Mahalanobis gating threshold
static constexpr int   MIN_EVENTS_TO_DRAW   =       10;   // draw only if ≥10 recent events
static constexpr size_t RECENT_BUFFER_SIZE  =    500;

using Event    = std::tuple<int,int,double>;  // x, y, timestamp(us)

using Color = SDL_Color;

struct EKFTrack {
    int id;
    Eigen::Matrix<double,7,1> x;   // [px, py, vx, vy, l1, l2, theta]
    Eigen::Matrix<double,7,7> P;   // covariance
    boost::circular_buffer<Event> recent;
    Eigen::Matrix2d LambdaInv;
    Eigen::Matrix2d R_obs;  // dynamic observation noise

    void setObservationNoise(const Eigen::Matrix2d &R) {
        R_obs = R;
    }

    EKFTrack(int _id, int x0, int y0, double t0)
      : id(_id), recent(RECENT_BUFFER_SIZE)
    {
        x << x0, y0, 0, 0, 10, 10, 0;
        P = Eigen::Matrix<double,7,7>::Identity() * 100.0;
        recent.push_back(Event{x0,y0,t0});
        updateLambdaInv();
        R_obs = Eigen::Matrix2d::Identity() * 0.5;
    }

    void addEvent(int ex, int ey, double et) {
        recent.push_back(Event{ex,ey,et});
        while (!recent.empty() && (et - std::get<2>(recent.front())) > PRUNE_WINDOW_US)
            recent.pop_front();
    }

    void predict(const Eigen::Matrix<double,7,7>& F, const Eigen::Matrix<double,7,7>& Q) {
        x = F * x;
        P = F * P * F.transpose() + Q;
        updateLambdaInv();
    }

    void updatePosition(double zx, double zy) {
        Eigen::Vector2d z(zx, zy);
        Eigen::Vector2d h = x.template segment<2>(0);
        Eigen::Matrix<double,2,7> H = Eigen::Matrix<double,2,7>::Zero();
        H(0,0)=1; H(1,1)=1;
        Eigen::Matrix2d R = R_obs;
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

    // Update inverse covariance based on current shape
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
    // time gate: must be updated recently
    if (trk.recent.empty() || (t - std::get<2>(trk.recent.back())) > GATE_TIME_US)
        return false;
    // Euclidean pre-gate
    double dx = double(x) - trk.x(0);
    double dy = double(y) - trk.x(1);
    if (dx*dx + dy*dy > MAX_ASSOC_DIST*MAX_ASSOC_DIST)
        return false;
    // Mahalanobis gate using track covariance inverse
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
    std::map<int, Color> color_map;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> color_dist(50,255);

    const double sampling_ratio =1;               // sample 10% of events for display
    std::uniform_real_distribution<double> uni(0,1);

    const int display_fps       = 30;                // draw at 10 FPS
    auto display_interval       = std::chrono::milliseconds(1000/display_fps);
    auto next_draw_time         = std::chrono::steady_clock::now() + display_interval;

    // For colored event sampling: store point and track id
    std::vector<std::pair<SDL_Point,int>> disp_events;
    disp_events.reserve(RECENT_BUFFER_SIZE);

    // --- 3) Main loop ---
    double last_time = std::get<2>(events.front());
    double start_time = last_time;
    // synchronize playback to real time
    auto wall_start = std::chrono::steady_clock::now();

    // SDL2 initialization for GPU-accelerated rendering
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    SDL_Window* sdl_window = SDL_CreateWindow("Tracking",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1280, 720, SDL_WINDOW_SHOWN);
    if (!sdl_window) {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }
    SDL_Renderer* sdl_renderer = SDL_CreateRenderer(sdl_window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!sdl_renderer) {
        std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(sdl_window);
        SDL_Quit();
        return 1;
    }
    
    for (auto &ev : events) {
        double predict_ms = 0.0, assoc_ms = 0.0, prune_ms = 0.0;

        double t = std::get<2>(ev);
        // Profiling start
        auto prof_loop_start = std::chrono::steady_clock::now();

        int ex = std::get<0>(ev), ey = std::get<1>(ev);
        // t already defined above

        // a) Predict existing tracks
        double dt = (t - last_time)*1e-6;
        last_time = t;

        // Precompute state transition and process noise matrices once per frame
        Eigen::Matrix<double,7,7> F_shared = Eigen::Matrix<double,7,7>::Identity();
        F_shared(0,2) = dt;
        F_shared(1,3) = dt;
        Eigen::Matrix<double,7,7> Q_shared = Eigen::Matrix<double,7,7>::Identity() * 1e-3;

        auto pred_start = std::chrono::steady_clock::now();
        for (auto &trk : active) trk.predict(F_shared, Q_shared);
        predict_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pred_start).count();

        // b) Associate / create tracks
        auto assoc_start = std::chrono::steady_clock::now();
        int assigned_id;
        double best_score = std::numeric_limits<double>::infinity();
        int best_idx = -1;

        // Global Nearest Neighbor association by Mahalanobis distance
        for (size_t i = 0; i < active.size(); ++i) {
            if (!mahalanobis_gate(active[i], ex, ey, t)) continue;
            // compute median-based centroid (same as gating)
            std::vector<double> xs, ys;
            xs.reserve(active[i].recent.size());
            ys.reserve(active[i].recent.size());
            for (const auto &ev : active[i].recent) {
                double ev_t = std::get<2>(ev);
                if ((t - ev_t) <= GATE_TIME_US) {
                    xs.push_back(std::get<0>(ev));
                    ys.push_back(std::get<1>(ev));
                }
            }
            if (xs.empty()) continue;
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            size_t m = xs.size() / 2;
            double cx = (xs.size() % 2) ? xs[m] : 0.5 * (xs[m - 1] + xs[m]);
            double cy = (ys.size() % 2) ? ys[m] : 0.5 * (ys[m - 1] + ys[m]);
            Eigen::Vector2d diff(double(ex) - cx, double(ey) - cy);
            double d2 = diff.transpose() * active[i].LambdaInv * diff;
            if (d2 < best_score) {
                best_score = d2;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0) {
            auto &trk = active[best_idx];
            trk.addEvent(ex, ey, t);
            // collect recent event coordinates into reusable buffers
            std::vector<double> xs_buf, ys_buf;
            xs_buf.reserve(trk.recent.size());
            ys_buf.reserve(trk.recent.size());
            for (const auto &ev : trk.recent) {
                double ev_t = std::get<2>(ev);
                if ((t - ev_t) <= GATE_TIME_US) {
                    xs_buf.push_back(std::get<0>(ev));
                    ys_buf.push_back(std::get<1>(ev));
                }
            }
            size_t N = xs_buf.size();
            if (N == 0) {
                assigned_id = trk.id;
            } else {
                // median-based centroid
                size_t m = N / 2;
                std::nth_element(xs_buf.begin(), xs_buf.begin() + m, xs_buf.end());
                std::nth_element(ys_buf.begin(), ys_buf.begin() + m, ys_buf.end());
                double cx = xs_buf[m];
                double cy = ys_buf[m];
                if (N % 2 == 0) {
                    double x1 = *std::max_element(xs_buf.begin(), xs_buf.begin() + m);
                    double y1 = *std::max_element(ys_buf.begin(), ys_buf.begin() + m);
                    cx = 0.5 * (x1 + xs_buf[m]);
                    cy = 0.5 * (y1 + ys_buf[m]);
                }
                // compute covariance and chi2
                double sum_xx = 0.0, sum_xy = 0.0, sum_yy = 0.0, chi2 = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    double dx = xs_buf[i] - cx;
                    double dy = ys_buf[i] - cy;
                    sum_xx += dx * dx;
                    sum_xy += dx * dy;
                    sum_yy += dy * dy;
                    chi2 += dx * dx + dy * dy;
                }
                Eigen::Matrix2d cov;
                cov << sum_xx / N, sum_xy / N,
                       sum_xy / N, sum_yy / N;
                trk.setObservationNoise(cov + Eigen::Matrix2d::Identity() * 1e-6);
                trk.updatePosition(cx, cy);
                trk.updateVariance(chi2);
                assigned_id = trk.id;
            }
        } else {
            active.emplace_back(++next_id, ex, ey, t);
            color_map[next_id] = { Uint8(color_dist(rng)), Uint8(color_dist(rng)), Uint8(color_dist(rng)), 255 };
            assigned_id = next_id;
        }

        assoc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - assoc_start).count();

        // --- f) Merge nearby tracks to avoid fragmentation ---
        for (size_t i = 0; i < active.size(); ++i) {
            for (size_t j = i + 1; j < active.size(); /* increment inside */) {
                // compute distance between centroids
                double dx = active[i].x(0) - active[j].x(0);
                double dy = active[i].x(1) - active[j].x(1);
                double dist2 = dx*dx + dy*dy;
                // compute velocity difference
                double dvx = active[i].x(2) - active[j].x(2);
                double dvy = active[i].x(3) - active[j].x(3);
                double vel_diff2 = dvx*dvx + dvy*dvy;
                if (dist2 <= MERGE_DIST * MERGE_DIST && vel_diff2 <= MERGE_VEL_DIFF * MERGE_VEL_DIFF) {
                    // merge j into i
                    // 1) merge recent buffers
                    for (auto &ev : active[j].recent) {
                        active[i].recent.push_back(ev);
                    }
                    // 2) average state vectors
                    active[i].x = 0.5 * (active[i].x + active[j].x);
                    // 3) average covariances
                    active[i].P = 0.5 * (active[i].P + active[j].P);
                    active[i].updateLambdaInv();
                    // erase track j
                    active.erase(active.begin() + j);
                    // do not increment j, new element now at j
                } else {
                    ++j;
                }
            }
        }

        // d) Sampling for display with track id
        if (uni(rng) < sampling_ratio) {
            disp_events.emplace_back(SDL_Point{ex, ey}, assigned_id);
        }

        // Profiling after tracking
        auto prof_after_track = std::chrono::steady_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(prof_after_track - prof_loop_start).count();

        // c) Prune old tracks
        auto prune_start = std::chrono::steady_clock::now();
        active.erase(
            std::remove_if(active.begin(), active.end(),
                [&](const EKFTrack &trk){
                    return (t - std::get<2>(trk.recent.back())) > PRUNE_WINDOW_US;
                }),
            active.end()
        );
        prune_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prune_start).count();

        // count events for this frame

        // Profiling before draw
        auto prof_before_draw = std::chrono::steady_clock::now();

        // e) Time to draw?
        auto now = std::chrono::steady_clock::now();
        if (now >= next_draw_time) {
            // frame-level playback sync to real time
            double sim_elapsed = (t - start_time) * 1e-6;
            auto wall_now     = std::chrono::steady_clock::now();
            double wall_elapsed = std::chrono::duration<double>(wall_now - wall_start).count();
            if (sim_elapsed > wall_elapsed) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(sim_elapsed - wall_elapsed)
                );
            }
            // detailed draw profiling start
            auto d0 = std::chrono::steady_clock::now();

            // clear and draw static background
            SDL_SetRenderDrawColor(sdl_renderer, 30, 30, 30, 255);
            SDL_RenderClear(sdl_renderer);

            // draw sampled event points by track color
            for (auto &e : disp_events) {
                Color c = color_map[e.second];
                SDL_SetRenderDrawColor(sdl_renderer, c.r, c.g, c.b, c.a);
                SDL_RenderDrawPoint(sdl_renderer, e.first.x, e.first.y);
            }

            // draw track centroids as small filled rectangles
            for (auto &trk : active) {
                if ((int)trk.recent.size() < MIN_EVENTS_TO_DRAW) continue;
                int cx = (int)trk.x(0), cy = (int)trk.x(1);
                SDL_Rect rect{cx - 2, cy - 2, 5, 5};
                Color c = color_map[trk.id];
                SDL_SetRenderDrawColor(sdl_renderer, c.r, c.g, c.b, c.a);
                SDL_RenderFillRect(sdl_renderer, &rect);
            }

            auto d3 = std::chrono::steady_clock::now();

            auto d4 = std::chrono::steady_clock::now();
            static int draw_prof_count = 0;
            if (++draw_prof_count % 30 == 0) {
                double t_bg   = std::chrono::duration<double, std::milli>(d3 - d0).count();
                double t_trk  = 0.0;
                double t_txt  = std::chrono::duration<double, std::milli>(d4 - d3).count();
                double t_show = 0.0;
                std::cout << "[DRAW] bg=" << t_bg << " ms, trk=" << t_trk
                          << " ms, text=" << t_txt << " ms, show=" << t_show << " ms\n";
            }

            // Profiling after draw
            auto prof_after_draw = std::chrono::steady_clock::now();
            double draw_ms = std::chrono::duration<double, std::milli>(prof_after_draw - prof_before_draw).count();
            static int prof_count = 0;
            if (++prof_count % 100 == 0) {
                std::cout << "[PERF] predict: " << predict_ms << " ms, associate: " << assoc_ms
                          << " ms, prune: " << prune_ms << " ms, total_track: " << track_ms
                          << " ms, draw: " << draw_ms << " ms\n";
            }

            // reset for next frame
            disp_events.clear();
            next_draw_time += display_interval;

            // render to window
            SDL_RenderPresent(sdl_renderer);

            // handle SDL quit event
            SDL_Event sdlevent;
            if (SDL_PollEvent(&sdlevent) && sdlevent.type == SDL_QUIT) break;
        }
    }

    // Cleanup SDL resources
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(sdl_window);
    SDL_Quit();
    // cv::waitKey(0);
    return 0;
}
