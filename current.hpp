#ifndef CURRENT_HPP
#define CURRENT_HPP

#include <armadillo>

#include "device.hpp"
#include "potential.hpp"

class current {
public:
    arma::vec lv;
    arma::vec rv;
    arma::vec lc;
    arma::vec rc;
    arma::vec lt;
    arma::vec rt;
    arma::vec total;

    inline current();
    inline current(const potential & phi);
    inline current(const wave_packet psi[6], const potential & phi0, const potential & phi);
};

//----------------------------------------------------------------------------------------------------------------------

current::current() {
}

current::current(const potential & phi)
    : lv(d::N_x), rv(d::N_x), lc(d::N_x), rc(d::N_x), lt(d::N_x), rt(d::N_x) {
    using namespace arma;
    
    // transmission probability
    auto transmission = [&] (double E) -> double {
        cx_double Sigma_s, Sigma_d;
        cx_vec G = green_col<false>(phi, E, Sigma_s, Sigma_d);
        return 4 * Sigma_s.imag() * Sigma_d.imag() * (std::norm(G(1)) + std::norm(G(2)));
    };
    
    static constexpr auto scale = 2.0 * c::e * c::e / c::h;
    
    vec E_lv, E_rv, E_lc, E_rc, E_lt, E_rt;
    vec W_lv, W_rv, W_lc, W_rc, W_lt, W_rt;
    
    auto i_lv = linspace(phi.s() + d::E_min, phi.s() - 0.5 * d::E_g, 50);
    auto i_rv = linspace(phi.d() + d::E_min, phi.d() - 0.5 * d::E_g, 50);
    auto i_lc = linspace(phi.s() + 0.5 * d::E_g, phi.s() + d::E_max, 50);
    auto i_rc = linspace(phi.d() + 0.5 * d::E_g, phi.d() + d::E_max, 50);
    
    lv.fill(integral<1>([&] (double E) {
        return - scale * transmission(E) * (1.0 - fermi(E - phi.s(), d::F_s));
    }, i_lv, d::rel_tol, c::epsilon(1e-10), E_lv, W_lv));
    
    rv.fill(integral<1>([&] (double E) {
        return scale * transmission(E) * (1.0 - fermi(E - phi.d(), d::F_d));
    }, i_rv, d::rel_tol, c::epsilon(1e-10), E_rv, W_rv));

    lc.fill(integral<1>([&] (double E) {
        return scale * transmission(E) * fermi(E - phi.s(), d::F_s);
    }, i_lc, d::rel_tol, c::epsilon(1e-10), E_lc, W_lc));
    
    rc.fill(integral<1>([&] (double E) {
        return - scale * transmission(E) * fermi(E - phi.d(), d::F_d);
    }, i_rc, d::rel_tol, c::epsilon(1e-10), E_rc, W_rc));
    
    if (phi.s() > phi.d() + d::E_g) {
        auto i_lt = linspace(phi.d() + 0.5 * d::E_g, phi.s() - 0.5 * d::E_g, 100);
        
        lt.fill(integral<1>([&] (double E) {
            return scale * transmission(E);
        }, i_lt, d::rel_tol, c::epsilon(1e-10), E_lt, W_lt));
        rt.fill(0.0);
    } else if (phi.d() > phi.s() + d::E_g) {
        auto i_rt = linspace(phi.s() + 0.5 * d::E_g, phi.d() - 0.5 * d::E_g, 100);
        
        lt.fill(0.0);
        rt.fill(integral<1>([&] (double E) {
            return scale * transmission(E);
        }, i_rt, d::rel_tol, c::epsilon(1e-10), E_rt, W_rt));
    } else {
        lt.fill(0.0);
        rt.fill(0.0);
    }
    
    total = lv + rv + lc + rc + lt + rt;
}

current::current(const wave_packet psi[6], const potential & phi0, const potential & phi)
    : lv(d::N_x), rv(d::N_x), lc(d::N_x), rc(d::N_x), lt(d::N_x), rt(d::N_x) {
    using namespace arma;
    
    // get imag(conj(psi) * psi)
    auto get_psi_I = [] (const cx_mat & m) {
        mat ret(m.n_rows / 2, m.n_cols);
        auto ptr0 = m.memptr();
        auto ptr1 = ret.memptr();
        for (unsigned i = 1; i < m.n_elem - 1; i += 2) {
            (*ptr1++) = std::imag(ptr0[i] * std::conj(ptr0[i + 1]));
        }
        for (unsigned i = 0; i < m.n_cols; ++i) {
            ret(d::N_x-1, i) = ret(d::N_x-2, i);
        }
        return ret;
    };
    auto psi_I_lv = get_psi_I(psi[LV].data);
    auto psi_I_rv = get_psi_I(psi[RV].data);
    auto psi_I_lc = get_psi_I(psi[LC].data);
    auto psi_I_rc = get_psi_I(psi[RC].data);

    static constexpr auto scale = 4.0 * d::t2 * c::e * c::e / c::h_bar / M_PI;

    lv = scale * psi_I_lv * psi[LV].W;
    rv = scale * psi_I_rv * psi[RV].W;
    lc = scale * psi_I_lc * psi[LC].W;
    rc = scale * psi_I_rc * psi[RC].W;
    lt.fill(0.0);
    rt.fill(0.0);

    if (psi[LT].E.size() > 0) {
        auto psi_I_lt = get_psi_I(psi[LT].data);
        unsigned i0;
        unsigned i1 = psi[LT].E.size() - 1;
        for (i0 = 0; i0 < i1; ++i0) {
            if ((psi[LT].E(i0) + phi.s() - phi0.s()) > phi.d()) {
                lt = scale * psi_I_lt.cols({i0, i1}) * psi[LT].W({i0, i1});
                break;
            }
        }
    }

    if (psi[RT].E.size() > 0) {
        auto psi_I_rt = get_psi_I(psi[RT].data);
        unsigned i0;
        unsigned i1 = psi[RT].E.size() - 1;
        for (i0 = 0; i0 < i1; ++i0) {
            if ((psi[RT].E(i0) + phi.d() - phi0.d()) > phi.s()) {
                rt = scale * psi_I_rt.cols({i0, i1}) * psi[RT].W({i0, i1});
                break;
            }
        }
    }

    total = lv + rv + lc + rc + lt + rt;
}

#endif
