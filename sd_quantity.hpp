#ifndef SD_QUANTITY_HPP
#define SD_QUANTITY_HPP

#include <armadillo>

class sd_vec {
public:
    arma::cx_vec s;
    arma::cx_vec d;

    inline sd_vec();
    inline sd_vec(int size);
};

sd_vec::sd_vec() {
}

sd_vec::sd_vec(int size)
    : s(size), d(size) {
}

class sd_mat {
public:
    arma::cx_mat s;
    arma::cx_mat d;

    inline sd_mat();
    inline sd_mat(int rows, int cols);
};

sd_mat::sd_mat() {
}

sd_mat::sd_mat(int rows, int cols)
    : s(rows, cols), d(rows, cols) {
}

#endif
