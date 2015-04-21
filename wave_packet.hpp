#ifndef WAVE_PACKET_HPP
#define WAVE_PACKET_HPP

#include <armadillo>

#include "constant.hpp"
#include "device.hpp"
#include "potential.hpp"
#include "time_params.hpp"
#include "sd_quantity.hpp"

enum {
    LV = 0,
    RV = 1,
    LC = 2,
    RC = 3,
    LT = 4,
    RT = 5
};

class wave_packet {
public:
    arma::vec E;
    arma::vec W;
    arma::cx_mat data;

    template<bool left>
    inline void init(const arma::vec & E, const arma::vec & W, const potential & phi);

    inline void memory_init();
    inline void memory_update(const sd_vec & affe, unsigned m);

    inline void source_init(const sd_vec & u, const sd_vec & q);
    inline void source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m);

    template<class T>
    inline void propagate(const T & U_eff, const sd_vec & inv);

    inline void remember();

    inline void update_sum(int m);

private:
    sd_vec in;
    sd_vec out;

    sd_mat sum;

    sd_vec source;
    sd_vec memory;

    // from previous time step
    arma::cx_mat old_data;
    sd_vec old_source;
};

//----------------------------------------------------------------------------------------------------------------------

template<bool left>
void wave_packet::init(const arma::vec & EE, const arma::vec & WW, const potential & phi) {
    using namespace arma;

    E = EE;
    W = WW;
    data = cx_mat(d::N_x * 2, E.size());
    in = sd_vec(E.size());
    out = sd_vec(E.size());
    sum = sd_mat(t::N_t, E.size());
    source = sd_vec(E.size());
    memory = sd_vec(E.size());

    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < E.size(); ++i) {
        // calculate 1 column of green's function
        cx_double Sigma_s, Sigma_d;
        cx_vec G = green_col<left>(phi, E(i), Sigma_s, Sigma_d);

        // calculate wave function
        if (left) {
            G *= std::sqrt(cx_double(- 2 * Sigma_s.imag()));
        } else {
            G *= std::sqrt(cx_double(- 2 * Sigma_d.imag()));
        }

        // extract data
        data.col(i) = G;
        in.s(i)  = G(0);
        in.d(i)  = G(G.size() - 1);

        // calculate first layer in the leads analytically
        out.s(i) = ((E(i) - phi.s()) * G(0) - d::t1 * G(1)) / d::t2;
        out.d(i) = ((E(i) - phi.d()) * G(G.size() - 1) - d::t1 * G(G.size() - 2)) / d::t2;
    }
}

void wave_packet::memory_init() {
    memory.s.fill(0.0);
    memory.d.fill(0.0);
}

void wave_packet::memory_update(const sd_vec & affe, unsigned m) {
    memory.s = (affe.s.st() * sum.s.rows({1, m - 1})).st();
    memory.d = (affe.d.st() * sum.d.rows({1, m - 1})).st();
}

void wave_packet::source_init(const sd_vec & u, const sd_vec & q) {
    using namespace std::complex_literals;
    source.s = - 2i * t::g * u.s(1) * (d::t2 * out.s + 1i * t::g * q.s(0) * in.s) / (1.0 + 1i * t::g * E);
    source.d = - 2i * t::g * u.d(1) * (d::t2 * out.d + 1i * t::g * q.d(0) * in.d) / (1.0 + 1i * t::g * E);
}

void wave_packet::source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m) {
    using namespace std::complex_literals;
    static constexpr auto g2 = t::g * t::g;

    source.s = (old_source.s % (1 - 1i * t::g * E) * u.s(m) * u.s(m-1) + 2 * g2 * L.s(1) / u.s(m) * qsum.s(t::N_t-m) * in.s) / (1 + 1i * t::g * E);
    source.d = (old_source.d % (1 - 1i * t::g * E) * u.d(m) * u.d(m-1) + 2 * g2 * L.d(1) / u.d(m) * qsum.d(t::N_t-m) * in.d) / (1 + 1i * t::g * E);
}

template<class T>
void wave_packet::propagate(const T & U_eff, const sd_vec & inv) {
    data = U_eff * old_data + arma::kron(source.s.st() + memory.s.st(), inv.s) + arma::kron(source.d.st() + memory.d.st(), inv.d);
}

void wave_packet::remember() {
    old_data = data;
    old_source = source;
}

void wave_packet::update_sum(int m) {
    sum.s.row(m) = old_data.row(0) + data.row(0);
    sum.d.row(m) = old_data.row(2*d::N_x-1) + data.row(2*d::N_x-1);
}

#endif
