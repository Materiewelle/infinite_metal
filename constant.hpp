#ifndef CONSTANT_HPP
#define CONSTANT_HPP

#include <armadillo>
#include <cmath>

namespace c {

    static constexpr auto eps_0 = 8.854187817E-12; // vacuum permittivity
    static constexpr auto e     = 1.602176565E-19; // elementary charge
    static constexpr auto h     = 6.62606957E-34;  // Planck constant
    static constexpr auto h_bar = h / 2 / M_PI;    // reduced Planck's constant
    static constexpr auto k_B   = 1.3806488E-23;   // Boltzmann constant
    static constexpr auto m_e   = 9.10938188E-31;  // electron mass
    static constexpr auto T     = 300;             // Temperature in Kelvin
    
    static inline constexpr auto epsilon(double x) {
        return std::nextafter(x, x + 1.0) - x;
    }

}

#endif

