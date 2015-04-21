#ifndef TIME_EVOLUTION_HPP
#define TIME_EVOLUTION_HPP

#include "anderson.hpp"
#include "gnuplot.hpp"
#include "sd_quantity.hpp"
#include "steady_state.hpp"
#include "time_params.hpp"
#include "voltage.hpp"
#include "wave_packet.hpp"

class time_evolution {
public:
    static constexpr auto dphi_threshold = 1e-8;
    static constexpr auto max_iterations = 50;
    static constexpr auto tunnel_current_precision = 5e-4;

    std::vector<current> I;
    std::vector<potential> phi;
    std::vector<charge_density> n;
    std::vector<voltage> V;

    inline time_evolution();
    inline time_evolution(const std::vector<voltage> & V);

    inline void solve();

private:
    sd_vec u;
    sd_vec L;
    sd_vec q;
    sd_vec qsum;

    template<bool left>
    void get_tunnel_energies(arma::vec & E, arma::vec & W);
    void calculate_q();
};

//----------------------------------------------------------------------------------------------------------------------

time_evolution::time_evolution()
    : I(t::N_t), phi(t::N_t), n(t::N_t), V(t::N_t), u(t::N_t), L(t::N_t), q(t::N_t), qsum(t::N_t - 1) {
}

time_evolution::time_evolution(const std::vector<voltage> & VV)
    : I(t::N_t), phi(t::N_t), n(t::N_t), V(VV), u(t::N_t), L(t::N_t), q(t::N_t), qsum(t::N_t - 1) {
}

void time_evolution::solve() {
    using namespace arma;
    using namespace std::complex_literals;

    // solve steady state
    steady_state s(V[0]);
    s.solve();

    // save results
    I[0]   = s.I;
    phi[0] = s.phi;
    n[0]   = s.n;

    // get tunnel energies
    arma::vec E_lt, E_rt;
    arma::vec W_lt, W_rt;
    get_tunnel_energies< true>(E_lt, W_lt);
    get_tunnel_energies<false>(E_rt, W_rt);

    // get initial wavefunctions
    wave_packet psi[6];
    psi[LV].init< true>(s.E[LV], s.W[LV], phi[0]);
    psi[RV].init<false>(s.E[RV], s.W[RV], phi[0]);
    psi[LC].init< true>(s.E[LC], s.W[LC], phi[0]);
    psi[RC].init<false>(s.E[RC], s.W[RC], phi[0]);
    psi[LT].init< true>(E_lt, W_lt, phi[0]);
    psi[RT].init<false>(E_rt, W_rt, phi[0]);

    // precalculate q-values
    calculate_q();

    // build constant part of Hamiltonian
    cx_mat H_eff(2*d::N_x, 2*d::N_x);
    H_eff.diag(+1) = conv_to<cx_vec>::from(d::t_vec);
    H_eff.diag(-1) = conv_to<cx_vec>::from(d::t_vec);

    anderson mr_neo;
    sd_vec affe;
    arma::cx_mat U_eff;
    sd_vec inv;
    sd_vec old_L;

    L.s.fill(1.0);
    L.d.fill(1.0);

    const cx_mat cx_eye = eye<cx_mat>(2 * d::N_x, 2 * d::N_x);

    // main loop of timesteps
    for (unsigned m = 1; m < t::N_t; ++m) {

        // estimate charge density from previous values
        n[m].data = (m == 1) ? n[m-1].data : (2 * n[m-1].data - n[m-2].data);

        // first guess for the potential
        phi[m] = potential(V[m], n[m]);
        mr_neo.reset(phi[m].data);

        // current data becomes old data
        for (int i = 0; i < 6; ++i) {
            psi[i].remember();
        }
        old_L = L;

        // self-consistency loop
        for (int it = 0; it < max_iterations; ++it) {
            // diagonal of H with self-energy
            H_eff.diag() = conv_to<cx_vec>::from(0.5 * (phi[m].twice + phi[m-1].twice));
            H_eff(         0,         0) -= 1i * t::g * q.s(0);
            H_eff(2*d::N_x-1,2*d::N_x-1) -= 1i * t::g * q.d(0);

            // crank-nicolson propagator
            U_eff = arma::solve(cx_eye + 1i * t::g * H_eff, cx_eye - 1i * t::g * H_eff);

            // inv
            inv.s = inverse_col< true>(cx_vec(1i * t::g * d::t_vec), cx_vec(1.0 + 1i * t::g * H_eff.diag()));
            inv.d = inverse_col<false>(cx_vec(1i * t::g * d::t_vec), cx_vec(1.0 + 1i * t::g * H_eff.diag()));

            // u
            u.s(m) = 0.5 * (phi[m].s() + phi[m - 1].s()) - phi[0].s();
            u.d(m) = 0.5 * (phi[m].d() + phi[m - 1].d()) - phi[0].d();
            u.s(m) = (1.0 - 0.5i * t::g * u.s(m)) / (1.0 + 0.5i * t::g * u.s(m));
            u.d(m) = (1.0 - 0.5i * t::g * u.d(m)) / (1.0 + 0.5i * t::g * u.d(m));

            // Lambda
            L.s({1, m}) = old_L.s({1, m}) * u.s(m) * u.s(m);
            L.d({1, m}) = old_L.d({1, m}) * u.d(m) * u.d(m);

            if (m == 1) {
                for (int i = 0; i < 4; ++i) {
                    psi[i].memory_init();
                    psi[i].source_init(u, q);
                    psi[i].propagate(U_eff, inv);
                }
            } else {
                affe.s = - t::g * t::g * L.s({1, m - 1}) % qsum.s({t::N_t-m, t::N_t-2}) / u.s({1, m - 1}) / u.s(m);
                affe.d = - t::g * t::g * L.d({1, m - 1}) % qsum.d({t::N_t-m, t::N_t-2}) / u.d({1, m - 1}) / u.d(m);

                // propagate wave functions of modes inside bands
                for (int i = 0; i < 4; ++i) {
                    psi[i].memory_update(affe, m);
                    psi[i].source_update(u, L, qsum, m);
                    psi[i].propagate(U_eff, inv);
                }
            }

            // update n
            n[m].update(psi);

            // update potential
            auto dphi = phi[m].update(V[m], n[m], mr_neo);

            cout << m << ": iteration " << it << ": rel deviation is " << dphi / dphi_threshold << endl;

            // check if dphi is small enough
            if (dphi < dphi_threshold) {
                break;
            }
        }

        // update wf in tunneling-region
        if (m == 1) {
            for (int i = LT; i <= RT; ++i) {
                psi[i].memory_init();
                psi[i].source_init(u, q);
                psi[i].propagate(U_eff, inv);
            }
        } else {
            for (int i = LT; i <= RT; ++i) {
                psi[i].memory_update(affe, m);
                psi[i].source_update(u, L, qsum, m);
                psi[i].propagate(U_eff, inv);
            }
        }

        // update sum
        for (int i = 0; i < 6; ++i) {
            psi[i].update_sum(m);
        }

        // calculate current
        I[m] = current(psi, phi[0], phi[m]);
    }
}

template<bool left>
void time_evolution::get_tunnel_energies(arma::vec & E, arma::vec & W) {
    double max = 0.0;
    double sgn = left ? 1.0 : -1.0;
    for (unsigned i = 0; i < V.size(); ++i) {
        double delta = (V[i].d + d::F_d - V[i].s - d::F_s) * sgn - 0.96 * d::E_g;
        if (delta > max) {
            max = delta;
        }
    }
    if (max > 0) {
        double E0 = (left ? (- V[0].s - d::F_s) : (- V[0].d - d::F_d)) - 0.48 * d::E_g;
        int N = std::round(max / tunnel_current_precision);
        E = arma::linspace(E0 - max, E0, N);
        W = arma::vec(N);
        W.fill(E(1)-E(0));
    } else {
        E = arma::vec(arma::uword(0));
        W = arma::vec(arma::uword(0));
    }
}

void time_evolution::calculate_q() {
    using namespace arma;
    using namespace std;
    using mat22 = cx_mat::fixed<2, 2>;

    // get q values dependent on potential in lead
    auto get_q = [&] (double phi0) {
        // shortcuts
        static constexpr auto t1 = d::t1;
        static constexpr auto t12 = t1 * t1;
        static constexpr auto t2 = d::t2;
        static constexpr auto t22 = t2 * t2;
        static constexpr auto g = t::g;
        static constexpr auto g2 = g * g;
        static const mat22 eye2 = { 1, 0, 0, 1 };

        // storage
        cx_vec qq(t::N_t + 3);
        vector<mat22> p(t::N_t + 3);

        // hamiltonian in lead
        mat22 h = { phi0, t1, t1, phi0 };

        // coupling hamiltonian
        mat22 Vau = { 0, t2, 0, 0 };

        // set first 3 values of q and p to 0
        for (int i = 0; i < 3; ++i) {
            qq(i) = 0;
            p[i] = { 0, 0, 0, 0 };
        }

        // first actual q value (wih pq-formula)
        auto a = (1.0 + 2i * g * phi0 + g2 * (t12 - t22 - phi0*phi0)) / g2 / (1.0 + 1i * g * phi0);
        qq(3) = - 0.5 * a + sqrt(0.25 * a * a + t22 / g2);

        // first actual p value
        p[3] = inv(eye2 + 1i * g * h + mat22({ g2 * qq(3), 0, 0, 0 }));

        // calculate A & C parameters
        mat22 A = eye2 + 1i * g * h + g2 * Vau.t() * p[3] * Vau;
        auto C = A(0,0) * A(1,1) - A(0,1) * A(1,0);

        // loop over all time steps
        for (int i = 4; i < t::N_t + 3; ++i) {
            // perform sum
            mat22 R = { 0, 0, 0, 0 };
            for (int k = 4; k < i; ++k) {
                R += (p[k] + 2 * p[k - 1] + p[k - 2]) * Vau * p[i - k + 3];
            }

            // calculate B parameter
            mat22 B = (eye2 - 1i * g * h) * p[i - 1] - g2 * Vau.t() * ((2 * p[i - 1] + p[i - 2]) * Vau * p[3] + R);

            // calculate next p values
            p[i](1,1) = (A(1,0) * B(0,1) - A(0,0) * B(1,1)) / (g2 * t22 * p[3](0,1) * A(1,0) - C);
            p[i](0,1) = (B(1,1) - A(1,1) * p[i](1,1)) / A(1,0);
            p[i](0,0) = (A(1,1) * B(0,0) - A(0,1) * B(1,0) - g2 * t22 * p[3](0,0) * p[i](1,1) * A(1,1)) / C;
            p[i](1,0) = (B(1,0) - A(1,0) * p[i](0,0)) / A(1,1);

            // calculate next q value
            qq(i) = t22 * p[i](1,1);
        }

        return qq;
    };

    // calculate and save q values
    q.s = get_q(phi[0].s())({3, t::N_t + 2});
    q.d = get_q(phi[0].d())({3, t::N_t + 2});

    // sum of two following q-values reversed
    for (int i = 0; i < t::N_t - 1; ++i) {
        qsum.s(i) = q.s(t::N_t - 1 - i) + q.s(t::N_t - 2 - i);
        qsum.d(i) = q.d(t::N_t - 1 - i) + q.d(t::N_t - 2 - i);
    }
}

#endif

