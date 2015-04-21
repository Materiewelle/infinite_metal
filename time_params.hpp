#ifndef TIME_PARAMS_HPP
#define TIME_PARAMS_HPP

#include "constant.hpp"

namespace t {

    static constexpr auto T   = 5e-15;                          // total simulated timespan
    static constexpr auto dt  = 1e-16;                          // timestep length
    static constexpr int  N_t = std::round(T / dt);             // number of steps
    static const     auto t   = arma::linspace(0, T - dt, N_t); // time lattice
    static constexpr auto g   = 0.5 * dt * c::e / c::h_bar;     // delta

}
#endif
