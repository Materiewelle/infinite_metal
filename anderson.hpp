#ifndef ANDERSON_HPP
#define ANDERSON_HPP

#include <armadillo>

class anderson {
public:
    static constexpr auto beta = 0.6;
    static constexpr auto N = 30;

    inline anderson();
    inline anderson(const arma::vec & v);
    inline void update(arma::vec & v, const arma::vec & f);
    inline void reset(const arma::vec & v);

private:
    arma::mat K;
    arma::mat D;
    arma::vec v_old;
    arma::vec f_old;
    int updates;
};

//----------------------------------------------------------------------------------------------------------------------

anderson::anderson() {
}

anderson::anderson(const arma::vec & v)
    : K(v.size(), 0), D(v.size(), 0), v_old(v), f_old(v.size()), updates(0) {
}

void anderson::update(arma::vec & v, const arma::vec & f) {
    using namespace arma;

    if (updates == 0) {
        v_old = v;
        f_old = f;

        v += f * beta;
    } else {
        K = join_horiz(K, v - v_old);
        D = join_horiz(D, f - f_old);

        while (D.n_cols > N) {
            K = K({0, K.n_rows - 1}, {1, K.n_cols - 1});
            D = D({0, D.n_rows - 1}, {1, D.n_cols - 1});
        }

        mat I = D.t() * D;

        while ((cond(I) > 1e16) && (D.n_cols > 1)) {
            K = K({0, K.n_rows - 1}, {1, K.n_cols - 1});
            D = D({0, D.n_rows - 1}, {1, D.n_cols - 1});
            I = D.t() * D;
        }

        v_old = v;
        f_old = f;

        v += f * beta - (K + D * beta) * I * D.t() * f;
    }

    ++updates;
}

void anderson::reset(const arma::vec & v) {
    K = arma::mat(v.size(), 0);
    D = arma::mat(v.size(), 0);
    v_old = arma::vec(v);
    f_old = arma::vec(v.size());
    updates = 0;
}

#endif
