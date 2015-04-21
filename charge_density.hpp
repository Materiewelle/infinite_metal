#ifndef CHARGE_DENSITY_HPP
#define CHARGE_DENSITY_HPP

#include <armadillo>

// forward declarations
#ifndef POTENTIAL_HPP
class potential;
#endif
#ifndef WAVE_PACKET_HPP
class wave_packet;
#endif

class charge_density {
public:
    static constexpr int initial_waypoints = 30;
    arma::vec data;

    inline charge_density();

    inline void update(const potential & phi, arma::vec E[4], arma::vec W[4]);
    inline void update(const wave_packet psi[4]);
};

// rest of includes
#include "constant.hpp"
#include "device.hpp"
#include "fermi.hpp"
#include "green.hpp"
#include "integral.hpp"
#include "potential.hpp"
#include "wave_packet.hpp"

//----------------------------------------------------------------------------------------------------------------------

namespace charge_density_impl {

    static inline arma::vec get_bound_states(const potential & phi);
    static inline arma::vec get_bound_states_interval(const potential & phi, double E_min, double E_max);

    template<bool source>
    static inline arma::vec get_A(const potential & phi, double E);

}

//----------------------------------------------------------------------------------------------------------------------

charge_density::charge_density()
    : data(d::N_x) {
    data.fill(0.0);
}

void charge_density::update(const potential & phi, arma::vec E[4], arma::vec W[4]) {
    using namespace arma;
    using namespace charge_density_impl;

    // get bound states
    auto E_bound = get_bound_states(phi);

    // get integration intervals
    auto get_intervals = [&] (double E_min, double E_max) {
        vec lin = linspace(E_min, E_max, initial_waypoints);

        if ((E_bound.size() > 0) && (E_bound(0) < E_max) && (E_bound(E_bound.size() - 1) > E_min)) {
            vec ret = vec(E_bound.size() + lin.size());

            // indices
            unsigned i0 = 0;
            unsigned i1 = 0;
            unsigned j = 0;

            // linear search, could be optimized to binary search
            while(E_bound(i1) < E_min) {
                ++i1;
            }

            // merge lin and E_bound
            while ((i0 < lin.size()) && (i1 < E_bound.size())) {
                if (lin(i0) < E_bound(i1)) {
                    ret(j++) = lin(i0++);
                } else {
                    ret(j++) = E_bound(i1++);
                }
            }

            // rest of lin, rest of E_bound out of range
            while(i0 < lin.size()) {
                ret(j++) = lin(i0++);
            }

            ret.resize(j);
            return ret;
        } else {
            return lin;
        }
    };
    vec i_sv = get_intervals(phi.s() + d::E_min, phi.s() - 0.5 * d::E_g);
    vec i_sc = get_intervals(phi.s() + 0.5 * d::E_g, phi.s() + d::E_max);
    vec i_dv = get_intervals(phi.d() + d::E_min, phi.d() - 0.5 * d::E_g);
    vec i_dc = get_intervals(phi.d() + 0.5 * d::E_g, phi.d() + d::E_max);

    // calculate charge density
    auto n_sv = integral<d::N_x>([&] (double E) -> vec {
        return get_A<true>(phi, E) * (fermi(E - phi.s(), d::F_s) - 1);
    }, i_sv, d::rel_tol, c::epsilon(1), E[LV], W[LV]);
    auto n_dv = integral<d::N_x>([&] (double E) -> vec {
        return get_A<false>(phi, E) * (fermi(E - phi.d(), d::F_d) - 1);
    }, i_dv, d::rel_tol, c::epsilon(1), E[RV], W[RV]);
    auto n_sc = integral<d::N_x>([&] (double E) -> vec {
        return get_A<true>(phi, E) * (fermi(E - phi.s(), d::F_s));
    }, i_sc, d::rel_tol, c::epsilon(1), E[LC], W[LC]);
    auto n_dc = integral<d::N_x>([&] (double E) -> vec {
        return get_A<false>(phi, E) * (fermi(E - phi.d(), d::F_d));
    }, i_dc, d::rel_tol, c::epsilon(1), E[RC], W[RC]);

    // multiply fermi function with weights
    for (unsigned i = 0; i < E[LV].size(); ++i) {
        W[LV](i) *= (1.0 - fermi(E[LV](i) - phi.s(), d::F_s));
    }
    for (unsigned i = 0; i < E[RV].size(); ++i) {
        W[RV](i) *= (1.0 - fermi(E[RV](i) - phi.d(), d::F_d));
    }
    for (unsigned i = 0; i < E[LC].size(); ++i) {
        W[LC](i) *= fermi(E[LC](i) - phi.s(), d::F_s);
    }
    for (unsigned i = 0; i < E[RC].size(); ++i) {
        W[RC](i) *= fermi(E[RC](i) - phi.d(), d::F_d);
    }

    // scaling factor
    static constexpr double scale = - c::e * 4 / M_PI / M_PI / d::dx / d::d_c / d::d_c;

    // scaling and doping
    data = (n_sv + n_sc + n_dv + n_dc) * scale + d::n0;
}

void charge_density::update(const wave_packet psi[4]) {
    using namespace arma;

    // get abs(psi)²
    auto get_abs = [] (const cx_mat & m) {
        mat ret(m.n_rows / 2, m.n_cols);
        auto ptr0 = m.memptr();
        auto ptr1 = ret.memptr();
        for (unsigned i = 0; i < m.n_elem; i += 2) {
            (*ptr1++) = std::norm(ptr0[i]) + std::norm(ptr0[i + 1]);
        }
        return ret;
    };

    vec n[4];
    for (int i = 0; i < 4; ++i) {
        n[i] = get_abs(psi[i].data) * psi[i].W;
    }

    // scaling factor
    static constexpr double scale = - c::e * 4 / M_PI / M_PI / d::dx / d::d_c / d::d_c;

    // scaling and doping
    data = (- n[LV] - n[RV] + n[LC] + n[RC]) * scale + d::n0;
}

arma::vec charge_density_impl::get_bound_states(const potential & phi) {
    double phi0, phi1, phi2, limit;

    // check for bound states in valence band
    phi0 = arma::min(phi.data(d::s)) - 0.5 * d::E_g;
    phi1 = arma::max(phi.data(d::c)) - 0.5 * d::E_g;
    phi2 = arma::min(phi.data(d::d)) - 0.5 * d::E_g;
    limit = phi0 > phi2 ? phi0 : phi2;
    if (limit < phi1) {
        return get_bound_states_interval(phi, limit, phi1);
    }

    // check for bound states in conduction band
    phi0 = arma::max(phi.data(d::s)) + 0.5 * d::E_g;
    phi1 = arma::min(phi.data(d::c)) + 0.5 * d::E_g;
    phi2 = arma::max(phi.data(d::d)) + 0.5 * d::E_g;
    limit = phi0 < phi2 ? phi0 : phi2;
    if (limit > phi1) {
        return get_bound_states_interval(phi, phi1, limit);
    }

    return arma::vec(arma::uword(0));
}

arma::vec charge_density_impl::get_bound_states_interval(const potential & phi, double E_min, double E_max) {
    using namespace arma;

    // return vector
    vec E_bound;

    // number of eigenvalues per step
    static constexpr int N = 20;

    // build the hamiltonian (without contacts)
    mat H = mat(d::N_x * 2, d::N_x * 2);
    H.fill(0);
    H.diag(-1) = d::t_vec;
    H.diag(+1) = d::t_vec;

    auto E_mid = E_min;

    while (true) {
        // eigenstates and -energies
        vec E;
        mat psi;

        // prepare hamiltonian so that eigenvalues near E_mid are searched
        H.diag(+0) = phi.twice - E_mid;

        // get eigenstates and -energies (sparse matrix created on the fly, probably inefficient)
        eigs_sym(E, psi, sp_mat(H), N, "sm"); // check if (always) sorted

        // reverse subtraction of E_mid
        E += E_mid;

        vec E2 = vec(E.size());
        unsigned N_E2 = 0;
        for (unsigned i = 0; i < E.size(); ++i) {
            // check if inside region of interest
            if ((E(i) > E_min) && (E(i) < E_max)) {
                // manual norm² since armadillo sucks
                double loc = 0;
                for (unsigned j = d::N_s; j < d::N_s + d::N_c; ++j) {
                    loc += psi(j, i) * psi(j, i);
                }

                // check if localized in the channel (more than sqrt(50%))
                if (loc >= 0.5) {
                    E2(N_E2++) = E(i);
                }
            }
        }

        if (N_E2 == 0) {
            sort(E_bound);
            return E_bound;
        } else {
            E2.resize(N_E2);
            E_bound = join_vert(E_bound, E2);
            E_mid = max(E2) + 0.4 * (max(E2) - min(E2));
            E_min = max(E2);
        }
    }
}

template<bool source>
arma::vec charge_density_impl::get_A(const potential & phi, const double E) {
    using namespace arma;

    // calculate 1 column of green's function
    cx_double Sigma_s, Sigma_d;
    cx_vec G = green_col<source>(phi, E, Sigma_s, Sigma_d);

    // get spectral function for each orbital (2 values per unit cell)
    vec A_twice;
    if (source) {
        A_twice = std::abs(2 * Sigma_s.imag()) * real(G % conj(G)); // G .* conj(G) = abs(G).^2
    } else {
        A_twice = std::abs(2 * Sigma_d.imag()) * real(G % conj(G));
    }

    // reduce spectral function to 1 value per unit cell (simple addition of both values)
    vec A = vec(d::N_x);
    for (unsigned i = 0; i < A.size(); ++i) {
        A(i) = A_twice(2 * i) + A_twice(2 * i + 1);
    }

    return A;
}

#endif

