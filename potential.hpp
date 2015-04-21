#ifndef POTENTIAL_HPP
#define POTENTIAL_HPP

#include <armadillo>

#include "anderson.hpp"
#include "voltage.hpp"

// forward declarations
#ifndef CHARGE_DENSITY_HPP
class charge_density;
#endif

class potential {
public:
    arma::vec data;
    arma::vec twice;

    inline potential();
    inline potential(const voltage & V);
    inline potential(const voltage & V, const charge_density & n);
    inline double update(const voltage & V, const charge_density & n, anderson & mr_neo);

    inline void smooth();

    inline double & operator()(int index);
    inline const double & operator()(int index) const;
    inline double s() const;
    inline double d() const;

private:
    template<bool minmax>
    inline void smooth(unsigned x0, unsigned x1);
    inline void update_twice();
};

// rest of includes
#include "charge_density.hpp"
#include "constant.hpp"
#include "device.hpp"

//----------------------------------------------------------------------------------------------------------------------

namespace potential_impl {

    static inline arma::vec poisson(const voltage & V);
    static inline arma::vec poisson(const voltage & V, const charge_density & n);
    static inline arma::vec get_R(const voltage & V);
    static inline arma::mat get_S();

    static const arma::mat S = get_S();

}

//----------------------------------------------------------------------------------------------------------------------

potential::potential()
    : twice(d::N_x * 2) {
}

potential::potential(const voltage & V)
    : twice(d::N_x * 2) {
    using namespace potential_impl;

    data = poisson(V);
    update_twice();
}

potential::potential(const voltage & V, const charge_density & n)
    : twice(d::N_x * 2) {
    using namespace potential_impl;

    data = poisson(V, n);
    update_twice();
}

double potential::update(const voltage & V, const charge_density & n, anderson & mr_neo) {
    using namespace arma;
    using namespace potential_impl;

    vec f = poisson(V, n) - data;

    // anderson mixing
    mr_neo.update(data, f);

    update_twice();

    // return dphi
    return max(abs(f));
}

void potential::smooth() {
    // smooth source region
    smooth<(d::F_s > 0)>(0, d::N_s + d::N_c * 0.2);

    // smooth drain region
    smooth<(d::F_d > 0)>(d::N_s + d::N_c * 0.8, d::N_x);

    update_twice();
}

double & potential::operator()(int index) {
    return data(index);
}
const double & potential::operator()(int index) const {
    return data(index);
}

double potential::s() const {
    return data(0);
}
double potential::d() const {
    return data(d::N_x - 1);
}

template<bool minmax>
void potential::smooth(unsigned x0, unsigned x1) {
    using namespace arma;

    if (minmax) {
        for (unsigned i = x0; i < x1 - 1; ++i) {
            if (data(i+1) >= data(i)) {
                continue;
            }
            for (unsigned j = i + 1; j < x1; ++j) {
                if (data(j) >= data(i)) {
                    data({i+1, j-1}).fill(data(i));
                    break;
                }
            }
        }
        for (unsigned i = x1 - 1; i >= x0 + 1; --i) {
            if (data(i-1) >= data(i)) {
                continue;
            }
            for (unsigned j = i - 1; j >= 1; --j) {
                if (data(j) >= data(i)) {
                    data({j+1, i-1}).fill(data(i));
                    break;
                }
            }
        }
    } else {
        for (unsigned i = x0; i < x1 - 1; ++i) {
            if (data(i+1) <= data(i)) {
                continue;
            }
            for (unsigned j = i + 1; j < x1; ++j) {
                if (data(j) <= data(i)) {
                    data({i+1, j-1}).fill(data(i));
                    break;
                }
            }
        }
        for (unsigned i = x1 - 1; i >= x0 + 1; --i) {
            if (data(i-1) <= data(i)) {
                continue;
            }
            for (unsigned j = i - 1; j >= 1; --j) {
                if (data(j) <= data(i)) {
                    data({j+1, i - 1}).fill(data(i));
                    break;
                }
            }
        }
    }
}

void potential::update_twice() {

    // duplicate each entry
    for (unsigned i = 0; i < d::N_x; ++i) {
        twice(2 * i    ) = data(i);
        twice(2 * i + 1) = data(i);
    }
}

arma::vec potential_impl::poisson(const voltage & V) {
    return solve(potential_impl::S, get_R(V));
}

arma::vec potential_impl::poisson(const voltage & V, const charge_density & n) {
    using namespace arma;

    // build right side
    auto R = get_R(V);
    R += n.data / c::eps_0 / d::eps_c * 1e9;

    return solve(potential_impl::S, R);
}

arma::vec potential_impl::get_R(const voltage & V) {
    using namespace arma;

    // build right side (without n)
    auto R = vec(d::N_x);
    R(d::s).fill((V.s + d::F_s) / d::lam_s / d::lam_s);
    R(d::c).fill((V.g + d::F_c) / d::lam_c / d::lam_c);
    R(d::d).fill((V.d + d::F_d) / d::lam_d / d::lam_d);

    return R;
}

arma::mat potential_impl::get_S() {
    using namespace arma;

    // main diagonal
    auto t0 = vec(d::N_x);
    t0(d::s).fill(- 2.0 / d::dx / d::dx - 1.0 / d::lam_s / d::lam_s);
    t0(d::c).fill(- 2.0 / d::dx / d::dx - 1.0 / d::lam_c / d::lam_c);
    t0(d::d).fill(- 2.0 / d::dx / d::dx - 1.0 / d::lam_d / d::lam_d);

    // off diagonal
    auto t1 = vec(d::N_x - 1);
    t1.fill(1.0 / d::dx / d::dx);

    // create matrix
    auto S = arma::mat(d::N_x, d::N_x);
    S.diag( 0) = t0;
    S.diag( 1) = t1;
    S.diag(-1) = t1;

    // von-Neumann boundary conditions
    S(         0,          1) *= 2;
    S(d::N_x - 1, d::N_x - 2) *= 2;

    return S;
}

#endif

