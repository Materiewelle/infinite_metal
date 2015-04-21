#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <armadillo>

#include "constant.hpp"
#include "fermi.hpp"
#include "integral.hpp"

namespace d {

    // material properties
    static constexpr auto eps_c = 11.2;                                             // relative permittivity of channel
    static constexpr auto eps_o = 12;                                               // relative permittivity of oxide
    static constexpr auto E_g   = 0.6;                                              // bandgap
    static constexpr auto m_eff = 0.2 * c::m_e;                                     // effective mass in conduction band
    static constexpr auto F_s   = -(E_g/2 + 0.0151);                                // Fermi level in source
    static constexpr auto F_c   = 0;                                                // Fermi level in channel
    static constexpr auto F_d   = +(E_g/2 + 0.0151);                                // Fermi level in drain

    // geometry (everything in nm)
    static constexpr auto l_c   = 10;                                               // channel length
    static constexpr auto d_c   = 2;                                                // channel thickness
    static constexpr auto d_o   = 0.8;                                              // oxide thickness
    static constexpr auto lam_c = sqrt(eps_c*d_c*d_c/8/eps_o * log(1 + 2*d_o/d_c)); // scr. length in channel
    static constexpr auto lam_s = 1.0 < lam_c ? 1.0 : lam_c;                        // scr. length in source
    static constexpr auto lam_d = 1.0 < lam_c ? 1.0 : lam_c;                        // scr. length in drain
    static constexpr auto l_s   = 15 * lam_s;                                       // source length
    static constexpr auto l_d   = 15 * lam_d;                                       // drain length
    static constexpr auto l     = l_s + l_c + l_d;                                  // device length

    // lattice
    static constexpr auto dx    = 0.1;                                              // lattice constant
    static constexpr int N_s    = round(l_s / dx);                                  // # of lattice points in source
    static constexpr int N_c    = round(l_c / dx);                                  // # of lattice points in channel
    static constexpr int N_d    = round(l_d / dx);                                  // # of lattice points in drain
    static constexpr auto N_x   = N_s + N_c + N_d;                                  // total # of lattice points
    static const     auto x     = arma::linspace(0.5 * dx, l - 0.5 * dx, N_x);      // lattice points

    // ranges
    static const     auto s     = arma::span(0, N_s - 1);
    static const     auto c     = arma::span(N_s, N_s + N_c - 1);
    static const     auto d     = arma::span(N_s + N_c, N_s + N_c + N_d - 1);

    // hopping parameters
    static constexpr auto t1    = 0.25 * E_g * (1 + sqrt(1 + 2 * c::h_bar*c::h_bar / (dx*dx * 1E-18 * m_eff * E_g * c::e)));
    static constexpr auto t2    = 0.25 * E_g * (1 - sqrt(1 + 2 * c::h_bar*c::h_bar / (dx*dx * 1E-18 * m_eff * E_g * c::e)));

    // off diagonal of hamiltonian
    template<int N>
    inline arma::vec create_t_vec() {
        arma::vec ret(N * 2 - 1);
        ret.imbue([&]() {
            static bool b = true;
            if (b) {
                b = false;
                return t1;
            } else {
                b = true;
                return t2;
            }
        });
        return ret;
    }
    static const     auto t_vec     = create_t_vec<N_x>();

    // integration parameters
    static constexpr auto E_min = -1.5;
    static constexpr auto E_max = +1.5;
    static constexpr auto rel_tol = 1e-3;

    // doping
    inline arma::vec create_n0() {
        using namespace arma;
        vec x0, x1, w0, w1;

        vec n0 = integral<3>([] (double E) {
            double dos = E / sqrt(4*t1*t1*t2*t2 - (E*E - t1*t1 - t2*t2) * (E*E - t1*t1 - t2*t2));
            vec ret = arma::vec(3);
            ret(0) = (1 - fermi(E, F_s)) * dos;
            ret(1) = (1 - fermi(E, F_c)) * dos;
            ret(2) = (1 - fermi(E, F_d)) * dos;
            return ret;
        }, linspace(E_min , - 0.5 * E_g, 100), rel_tol, std::numeric_limits<double>::epsilon(), x0, w0) + integral<3>([] (double E) {
            double dos = E / sqrt(4*t1*t1*t2*t2 - (E*E - t1*t1 - t2*t2) * (E*E - t1*t1 - t2*t2));
            vec ret = arma::vec(3);
            ret(0) = fermi(E, F_s) * dos;
            ret(1) = fermi(E, F_c) * dos;
            ret(2) = fermi(E, F_d) * dos;
            return ret;
        }, linspace(0.5 * E_g, E_max, 100), rel_tol, std::numeric_limits<double>::epsilon(), x1, w1);

        vec ret(N_x);
        ret(s).fill(n0(0));
        ret(c).fill(n0(1));
        ret(d).fill(n0(2));

        ret *= 16 * c::e / M_PI / M_PI / dx / d_c / d_c;

        return ret;
    }
    static const auto n0 = create_n0();

}

#endif

