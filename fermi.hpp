#ifndef FERMI_HPP
#define FERMI_HPP

#include "constant.hpp"

inline double fermi(double E, double F) {
    return 1.0 / (1.0 + std::exp((E - F) * c::e / c::T / c::k_B));
}

#endif
