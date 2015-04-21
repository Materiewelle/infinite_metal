#ifndef STEADY_STATE_HPP
#define STEADY_STATE_HPP

#include <armadillo>

#include "anderson.hpp"
#include "charge_density.hpp"
#include "current.hpp"
#include "potential.hpp"
#include "voltage.hpp"

class steady_state {
public:
    static constexpr auto dphi_threshold = 1e-12;
    static constexpr auto max_iterations = 250;

    voltage V;
    charge_density n;
    potential phi;
    current I;
    arma::vec E[4];
    arma::vec W[4];

    inline steady_state(const voltage & V);
    inline steady_state(const voltage & V, const potential & phi0);

    template<bool smooth = true>
    inline bool solve();

    static inline void output(const voltage & V0, double V_d1, int N, arma::vec & V_d, arma::vec & I);
    static inline void transfer(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & I);
};

//----------------------------------------------------------------------------------------------------------------------

steady_state::steady_state(const voltage & VV)
    : V(VV), n(), phi(V) {
}

steady_state::steady_state(const voltage & VV, const potential & phi0)
    : V(VV), n(), phi(phi0) {
}

template<bool smooth>
bool steady_state::solve() {
    using namespace std;

    // dphi = norm(delta_phi)
    double dphi;

    // iteration counter
    int it;

    anderson mr_neo(phi.data);

    // repeat until potential does not change anymore or iteration limit has been reached
    for (it = 1; it <= max_iterations; ++it) {
        // update charge density
        n.update(phi, E, W);

        // update potential
        dphi = phi.update(V, n, mr_neo);

        cout << V.s << ", " << V.g << ", " << V.d;
        cout << ": iteration " << it << ": rel deviation is " << dphi/dphi_threshold << endl;

        // check if dphi is small enough
        if (dphi < dphi_threshold) {
            break;
        }

        if (smooth) {
            // smooth potential in the beginning
            if (it < 3) {
                phi.smooth();
            }
        }
    }

    // get current
    I = current(phi);

    // check if actually converged
    if (dphi > dphi_threshold) {
        cout << "Warning: steady_state::solve did not converge after " << it << " iterations!" << endl;
        return false;
    } else {
        return true;
    }
}

void steady_state::output(const voltage & V0, double V_d1, int N, arma::vec & V_d, arma::vec & I) {
    V_d = arma::linspace(V0.d, V_d1, N);
    I = arma::vec(N);

    steady_state s(V0);
    bool conv = s.solve();
    I(0) = s.I.total(0);

    for (int i = 1; i < N; ++i) {
        voltage V = { V0.s, V0.g, V_d(i) };
        if (conv) {
            s = steady_state(V, s.phi);
            conv = s.solve<false>();
        } else {
            s = steady_state(V);
            conv = s.solve();
        }
        I(i) = s.I.total(0);
    }
}

void steady_state::transfer(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & I) {
    V_g = arma::linspace(V0.g, V_g1, N);
    I = arma::vec(N);

    steady_state s(V0);
    bool conv = s.solve();
    I(0) = s.I.total(0);

    for (int i = 0; i < N; ++i) {
        voltage V = { V0.s, V_g(i), V0.d };
        if (conv) {
            s = steady_state(V, s.phi);
            conv = s.solve<false>();
        } else {
            s = steady_state(V);
            conv = s.solve();
        }
        I(i) = s.I.total(0);
    }
}

#endif

