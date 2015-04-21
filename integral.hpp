#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#include <armadillo>
#include <queue>

template<int N, class F>
inline auto integral(F && f, const arma::vec & intervals, double tol, double tol_f, arma::vec & x, arma::vec & w);

namespace integral_impl {

    using namespace arma;

    template<class T>
    inline auto correct_abs(T x) {
        return abs(x);
    }
    template<>
    inline auto correct_abs<>(double x) {
        return std::abs(x);
    }
    
    template<int N>
    struct ret_type {
        using type = vec;
        using y_type = mat;
        static inline auto construct_type(double init) {
            vec ret(N);
            ret.fill(init);
            return ret;
        }
        static inline auto construct_y_type(int M) {
            return mat(N, M);
        }
        static inline void fill_y(mat & y, int i, vec && f) {
            y.col(i) = f;
        }
        static inline bool is_finite_y(mat & y, int i) {
            return y.col(i).is_finite();
        }
        static inline mat select_y(mat & y, unsigned i0, unsigned i1) {
            return y.cols({i0, i1});
        }
        static inline void resize_y(mat & y, int M) {
            y.resize(N, M);
        }
        static inline vec estimate(mat & y, int i0, int i1, int i2) {
            return (y.col(i0) + 4 * y.col(i1) + y.col(i2)) / 6;
        }
        static inline bool convergence(vec & delta_I, vec & I2_abs, double tol) {
            return all(abs(delta_I) <= tol * I2_abs);
        }
    };
    template<>
    struct ret_type<1> {
        using type = double;
        using y_type = vec;
        static inline auto construct_type(double init) {
            return init;
        }
        static inline auto construct_y_type(int M) {
            return vec(M);
        }
        static inline void fill_y(vec & y, int i, double f) {
            y(i) = f;
        }
        static inline bool is_finite_y(vec & y, int i) {
            return is_finite(y(i));
        }
        static inline double select_y(vec & y, int i) {
            return y(i);
        }
        static inline vec select_y(vec & y, unsigned i0, unsigned i1) {
            return y({i0, i1});
        }
        static inline void resize_y(vec & y, int M) {
            y.resize(M);
        }
        static inline double estimate(vec & y, int i0, int i1, int i2) {
            return (y(i0) + 4 * y(i1) + y(i2)) / 6;
        }
        static inline bool convergence(double delta_I, double I2_abs, double tol) {
            return std::abs(delta_I) <= tol * I2_abs;
        }
    };
    
    template<int N>
    struct interval_data {
        unsigned index[5];
        typename ret_type<N>::type I;

        inline unsigned operator()(int i) const {
            return index[i];
        }
    };

    template<int N, class F>
    inline auto integral_interval(F && f, vec & x, typename ret_type<N>::y_type & y, vec & w, double tol, double tol_f);

}

//----------------------------------------------------------------------------------------------------------------------

template<int N, class F>
auto integral(F && f, const arma::vec & intervals, double tol, double tol_f, arma::vec & x, arma::vec & w) {
    using namespace arma;
    using namespace integral_impl;

    // initial x values
    vec x0 = vec(4 * intervals.size() - 3);
    x0(0) = intervals(0);
    for (unsigned i = 1; i < intervals.size(); ++i) {
        x0(4 * i - 3) = 0.75 * intervals(i - 1) + 0.25 * intervals(i);
        x0(4 * i - 2) = 0.50 * intervals(i - 1) + 0.50 * intervals(i);
        x0(4 * i - 1) = 0.25 * intervals(i - 1) + 0.75 * intervals(i);
        x0(4 * i - 0) = 0.00 * intervals(i - 1) + 1.00 * intervals(i);
    }

    // matrix / vec to hold initial y values;
    auto y0 = ret_type<N>::construct_y_type(x0.size());

    // integral value
    auto I = ret_type<N>::construct_type(0);

    // x_i and w_i vectors
    vec x_i[intervals.size() - 1];
    vec w_i[intervals.size() - 1];

    // total number of x points
    unsigned x_n = 0;

    // enter multithreaded region
    #pragma omp parallel reduction(+ : x_n)
    {
        // local variable for intermediate integral value
        auto I_thread = ret_type<N>::construct_type(0);

        // calculate y values for initial x points
        #pragma omp for schedule(static)
        for (unsigned i = 0; i < x0.size(); ++i) {
            ret_type<N>::fill_y(y0, i, f(x0(i)));
            while(!ret_type<N>::is_finite_y(y0, i)) {
                x0(i) = std::nextafter(x0(i), x0(i) + std::copysign(1.0, x0(i)));
                ret_type<N>::fill_y(y0, i, f(x0(i)));
            }
        }
        // implied omp barrier

        // integrate each interval; schedule(dynamic) since each interval takes a different amount of time to compute
        #pragma omp for schedule(dynamic)
        for (unsigned i = 0; i < intervals.size() - 1; ++i) {
            x_i[i] = x0({4 * i, 4 * i + 4});
            w_i[i] = vec(5);
            auto y1 = ret_type<N>::select_y(y0, 4 * i, 4 * i + 4);

            I_thread += integral_interval<N>(f, x_i[i], y1, w_i[i], tol, tol_f);
            x_n += x_i[i].size();
        }
        // implied omp barrier

        // perform reduction of intermediate integral values
        #pragma omp critical
        {
            I += I_thread;
        }
    }

    // merge x_i and w_i vectors
    x.resize(x_n - (intervals.size() - 2));
    w.resize(x_n - (intervals.size() - 2));
    unsigned i0 = 0;
    unsigned i1 = x_i[0].size();
    x({i0, i1 - 1}) = x_i[0];
    w({i0, i1 - 1}) = w_i[0];
    for (unsigned i = 1; i < intervals.size() - 1; ++i) {
        i0 = i1;
        i1 += x_i[i].size() - 1;
        x({i0, i1 - 1}) = x_i[i]({1, x_i[i].size() - 1});
        w({i0, i1 - 1}) = w_i[i]({1, w_i[i].size() - 1});

        // add overlapping weights
        w(i0 - 1) += w_i[i](0);
    }

    return I;
}

namespace integral_impl {

    template<int N, class F>
    auto integral_interval(F && f, vec & x, typename ret_type<N>::y_type & y, vec & w, double tol, double tol_f) {
        std::queue<interval_data<N>> queue;
        double min_h = c::epsilon(x(4) - x(0)) / 1024.0;
        bool stop_it = false;

        // return value
        auto I = ret_type<N>::construct_type(0);

        // preallocation
        unsigned n = 5;
        x.resize(25);
        ret_type<N>::resize_y(y, 25);
        w.resize(25);

        // weights
        w.fill(0.0);

        // initial simpson estimate on rough grid
        typename ret_type<N>::type I1 = (x(4) - x(0)) * ret_type<N>::estimate(y, 0, 2, 4);

        // put initial data on stack
        queue.push({{0, 1, 2, 3, 4}, I1});

        // loop until no more data left
        while (!queue.empty()) {
            // too many points ?
            if (x.size() > 5000) {
                stop_it = true;
            }

            // select interval
            interval_data<N> & i = queue.front();

            // interval size
            double h = x(i(4)) - x(i(0));

            // simpson estimate on fine grid
            typename ret_type<N>::type I2l = 0.5 * h * ret_type<N>::estimate(y, i(0), i(1), i(2));
            typename ret_type<N>::type I2r = 0.5 * h * ret_type<N>::estimate(y, i(2), i(3), i(4));
            auto I2  = I2l + I2r;
            typename ret_type<N>::type I2_abs = correct_abs(I2);

            // difference between two estimates
            typename ret_type<N>::type delta_I = I2 - i.I;

            // convergence condition
            if ((ret_type<N>::convergence(delta_I, I2_abs, tol)) || (max(I2_abs) <= tol_f) || (h <= min_h) || stop_it) {
                I += I2 + delta_I / 15;

                // update weights
                w(i(0)) += h *  7 / 90;
                w(i(1)) += h * 16 / 45;
                w(i(2)) += h *  2 / 15;
                w(i(3)) += h * 16 / 45;
                w(i(4)) += h *  7 / 90;

            } else {
                // check if capacity of vectors is sufficient
                if (x.size() < n + 4) {
                    unsigned capacity = x.size() * 2;
                    x.resize(capacity);
                    ret_type<N>::resize_y(y, capacity);
                    w.resize(capacity);
                }

                // indices
                unsigned j0 = n;
                unsigned j1 = n + 4;

                // update size
                n += 4;

                // eval function at new points (4 evals)
                for (unsigned j = j0; j < j1; ++j) {
                    x(j) = h / 8 + x(i(j - j0));
                    ret_type<N>::fill_y(y, j, f(x(j)));
                    while(!ret_type<N>::is_finite_y(y, j)) {
                        x(j) = std::nextafter(x(j), x(j) + std::copysign(1.0, x(j)));
                        ret_type<N>::fill_y(y, j, f(x(j)));
                    }
                    w(j) = 0;
                }

                // put two new intervals on the queue
                queue.push({{i(0), j1 - 4, i(1), j1 - 3, i(2)}, I2l});
                queue.push({{i(2), j1 - 2, i(3), j1 - 1, i(4)}, I2r});
            }

            // remove first element from queue
            queue.pop();
        }

        // cut off excess space
        x.resize(n);
        w.resize(n);

        // sort x and w
        // save sort_index as uvec not auto since armadillo treats it as glue => sorting of w breaks after x is sorted
        uvec s = sort_index(x);
        x = x(s);
        w = w(s);

        return I;
    }

}

#endif
