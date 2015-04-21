//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>
#include "time_evolution.hpp"
#include "green.hpp"

using namespace arma;
using namespace std;

int main() {

    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    time_evolution te;
//    std::fill(begin(te.V), begin(te.V) + 20, voltage{0.0, 0.0, 0.8});
//    for (int i = 20; i < 50; ++i) {
//        te.V[i] = { 0.0, 0.0 + 0.5 * double(i - 20) / 30.0, 0.8};
//    }
//    std::fill(begin(te.V) + 50, end(te.V), voltage{0.0, 0.5, 0.8});



    std::fill(begin(te.V), end(te.V), voltage{0.0, -0.3, 0.8});

    wall_clock timer;

    timer.tic();

    te.solve();

//    cout << te.I[0].lv(0) << endl;
//    cout << te.I[0].rv(0) << endl;
//    cout << te.I[0].lc(0) << endl;
//    cout << te.I[0].rc(0) << endl;
//    cout << te.I[0].lt(0) << endl;
//    cout << te.I[0].rt(0) << endl;
//    cout << endl;
//    cout << te.I[1].lv(0) << endl;
//    cout << te.I[1].rv(0) << endl;
//    cout << te.I[1].lc(0) << endl;
//    cout << te.I[1].rc(0) << endl;
//    cout << te.I[1].lt(0) << endl;
//    cout << te.I[1].rt(0) << endl;
//    cout << endl;
//    cout << te.I[1].lt(0) / te.I[0].lt(0) << endl;
    cout << timer.toc() << endl;

//    plot(te.I[t::N_t-1].total);
//    plot(s.I.total);

//    steady_state s({0., 0., 0.});
//    vec E_grid;
//    mat ldos;
//    ldos = lDOS(s.phi, 3000, E_grid);
//    gnuplot gp;
//    gp << "set terminal pdf rounded enhanced font 'arial,12'\n";
//    gp << "set output \"lDOS.pdf\"\n";
//    gp << "set xlabel \"x / nm\"\n";
//    gp << "set ylabel \"E / eV\"\n";
//    gp << "set title \"TFET logarithmic lDOS\"\n";
//    gp.set_background(d::x, E_grid, ldos);
//    gp.plot();

//    gnuplot gp1;
//    gp1 << "set title \"te.phi\"\n";
//    gp1.add(d::x, te.phi[t::N_t-1].data);
//    gp1.plot();
//    gnuplot gp1a;
//    gp1a << "set title \"te.n\"\n";
//    gp1a.add(d::x, te.n[t::N_t-1].data);
//    gp1a.plot();
//    gnuplot gp2;
//    gp2 << "set title \"te.I\"\n";
//    gp2.add(d::x, te.I[t::N_t-1].total);
//    gp2.plot();

//    gnuplot gp3;
//    gp3 << "set title \"s.phi\"\n";
//    gp3.add(d::x, s.phi.data);
//    gp3.plot();
//    gnuplot gp3a;
//    gp3a << "set title \"s.n\"\n";
//    gp3a.add(d::x, s.n.data);
//    gp3a.plot();
//    gnuplot gp4;
//    gp4 << "set title \"s.I\"\n";
//    gp4.add(d::x, s.I.total);
//    gp4.plot();

    return 0;
}
