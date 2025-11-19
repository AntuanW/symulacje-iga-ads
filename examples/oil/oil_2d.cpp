#include "oil2d.hpp"


int main() {
    ads::dim_config dim{2, 20};
    ads::timesteps_config steps{10000, 1e-7};
    int ders = 1;

    ads::config_2d c{dim, dim, steps, ders};
    ads::oil2d sim{c};
    sim.run();
}