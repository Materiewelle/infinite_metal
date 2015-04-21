#ifndef GREEN_HPP
#define GREEN_HPP

#include <armadillo>
#include <complex>

#include "constant.hpp"
#include "device.hpp"
#include "inverse.hpp"
#include "potential.hpp"
#include "gnuplot.hpp"

static inline void self_energy(const potential & phi, double E, arma::cx_double & Sigma_s, arma::cx_double & Sigma_d) {
    using namespace arma;
    using namespace std;
    // kinetic energy in source and drain
    auto E_s = E - phi.s();
    auto E_d = E - phi.d();

    // get wave vectors (times lattice constant)
    auto k_s = acos((E_s * E_s - d::t1 * d::t1 - d::t2 * d::t2) / (2 * d::t1 * d::t2) + 0i);
    auto k_d = acos((E_d * E_d - d::t1 * d::t1 - d::t2 * d::t2) / (2 * d::t1 * d::t2) + 0i);
    k_s = copysign(1.0, E_s) * k_s.real() + abs(k_s.imag()) * 1i;
    k_d = copysign(1.0, E_d) * k_d.real() + abs(k_d.imag()) * 1i;

    // self energy
    Sigma_s = (d::t1 * d::t2 * exp(1i * k_s) + d::t2 * d::t2) / E_s;
    Sigma_d = (d::t1 * d::t2 * exp(1i * k_d) + d::t2 * d::t2) / E_d;
}

template<bool source>
static inline arma::cx_vec green_col(const potential & phi, double E, arma::cx_double & Sigma_s, arma::cx_double & Sigma_d) {
    using namespace arma;

    static const arma::vec t_vec_neg     = - d::t_vec;

    self_energy(phi, E, Sigma_s, Sigma_d);

    // build diagonal part of hamiltonian
    auto D = conv_to<cx_vec>::from(E - phi.twice);
    D(0)            -= Sigma_s;
    D(D.size() - 1) -= Sigma_d;

    return inverse_col<source>(t_vec_neg, D);
}

static inline arma::mat ldos(const potential & phi, int N_grid, arma::vec & E) {
    using namespace arma;
    using namespace std::complex_literals;

    static const arma::vec t_vec_neg     = - d::t_vec;

    mat ret(N_grid, d::N_x);

    double phi_min = min(phi.data);
    double phi_max = max(phi.data);

    E = linspace(phi_min - 0.5 * d::E_g - 0.2, phi_max + 0.5 * d::E_g + 0.2, N_grid);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_grid; ++i) {
        cx_double Sigma_s;
        cx_double Sigma_d;
        self_energy(phi, E(i), Sigma_s, Sigma_d);

        auto D = conv_to<cx_vec>::from(E(i) - phi.twice);
        D(0)            -= Sigma_s;
        D(D.size() - 1) -= Sigma_d;
        D += 0.001i;

        vec mixed = -arma::imag(inverse_diag(t_vec_neg, D)) / M_PI;

        for (int j = 0; j < d::N_x; ++j) {
            ret(i, j) = mixed(2*j) + mixed(2*j+1);
        }
    }
    return ret;
}

static void plot_ldos(const potential & phi, const unsigned N_grid) {
    gnuplot gp;

    gp << "set title \"Logarithmic lDOS\"\n";
    gp << "set xlabel \"x / nm\"\n";
    gp << "set ylabel \"E / eV\"\n";
    gp << "set zlabel \"log(lDOS)\"\n";
    gp << "unset key\n";
    gp << "unset colorbox\n";

    arma::vec E;
    arma::mat lDOS = ldos(phi, N_grid, E);
    gp.set_background(d::x, E, arma::log(lDOS));

    gp.add(d::x, phi.data + 0.5 * d::E_g);
    gp.add(d::x, phi.data - 0.5 * d::E_g);

    unsigned N_s = std::round(0.5 * d::N_s);
    arma::vec fermi_l(N_s);
    fermi_l.fill(d::F_s + phi.s());
    arma::vec x_l = d::x(arma::span(0, N_s-1));
    gp.add(x_l, fermi_l);

    unsigned N_d = std::round(0.5 * d::N_d);
    arma::vec fermi_r(N_d);
    fermi_r.fill(d::F_d + phi.d());
    arma::vec x_r = d::x(arma::span(d::N_x-N_d, d::N_x-1));
    gp.add(x_r, fermi_r);

    gp << "set style line 1 lt 1 lc rgb '#F6A800' lw 2\n";
    gp << "set style line 2 lt 1 lc rgb '#F6A800' lw 2\n";
    gp << "set style line 3 lt 3 lc rgb '#000000' lw 1\n";
    gp << "set style line 4 lt 3 lc rgb '#000000' lw 1\n";

    gp.plot();
}

#endif

