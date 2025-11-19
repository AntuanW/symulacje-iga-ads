#ifndef OIL2D_HPP
#define OIL2D_HPP

#include <cmath>

#include "ads/executor/galois.hpp"
#include "ads/output_manager.hpp"
#include "ads/simulation.hpp"

namespace ads {

struct vec2d {
    double x;
    double y;
};

inline vec2d operator-(const vec2d& a, const vec2d& b) {
    return {a.x - b.x, a.y - b.y};
}

inline vec2d operator+(const vec2d& b, const vec2d& a) {
    return {a.x + b.x, a.y + b.y};
}

inline vec2d operator*(double c, const vec2d& a) {
    return {c * a.x, c * a.y};
}

inline double dot(const vec2d& a, const vec2d& b) {
    return a.x * b.x + a.y * b.y;
}

inline double len_sq(const vec2d& a) {
    return dot(a, a);
}

inline double len(const vec2d& a) {
    return std::sqrt(len_sq(a));
}

inline double falloff(double r, double R, double t) {
    if (t < r)
        return 1.0;
    if (t > R)
        return 0.0;
    double h = (t - r) / (R - r);
    return std::pow((h - 1) * (h + 1), 2);
}

// r < R in [0, 1]
inline double bump(double r, double R, double x, double y) {
    double dx = x - 0.5;
    double dy = y - 0.5;
    double t = std::sqrt(dx * dx + dy * dy);
    return falloff(r / 2, R / 2, t);
}

struct pumps {
    std::vector<ads::vec2d> sources;
    std::vector<ads::vec2d> sinks;

    static constexpr double radius = 0.15;
    static constexpr double pumping_strength = 1;
    static constexpr double draining_strength = 1e5;

    double pumping(double x, double y) const {
        ads::vec2d v{x, y};
        double p = 0;
        for (const auto& pos : sources) {
            double dist = len(v - pos);
            p += pumping_strength * ads::falloff(0, radius, dist);
        }
        return p;
    }

    double draining(double u, double x, double y) const {
        ads::vec2d v{x, y};
        double p = 0;
        for (const auto& pos : sinks) {
            double dist = len(v - pos);
            double s = draining_strength * ads::falloff(0, radius, dist);
            p += u * s;
        }
        return p;
    }

    auto pumping_fun() const {
        return [this](double x, double y) { return pumping(x, y); };
    }
};

class oil2d : public simulation_2d {
private:
    using Base = simulation_2d;
    vector_type u, u_prev;

    galois_executor executor{4};

    pumps process = pumps{{{0.25, 0.25}, {0.75, 0.75}}, {{0.25, 0.75}, {0.75, 0.25}}};
    lin::tensor<double, 4> kq;
    output_manager<2> output;

public:
    explicit oil2d(const config_2d& config)
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , kq{{x.basis.elements, y.basis.elements, x.basis.quad_order + 1, y.basis.quad_order + 1}}
    , output{x.B, y.B, 100} { }

    double init_state(double x, double y) {
        double r = 0.1;
        double R = 0.5;
        return 1e-3 * ads::bump(r, R, x, y);
    };

private:
    void before() override {
        fill_permeability_map();
        prepare_matrices();

        auto init = [this](double x, double y) { return init_state(x, y); };
        projection(u, init);
        solve(u);
        output.to_file(u, "out_%d.data", 0);
    }

    void fill_permeability_map() {
        for (auto e : elements()) {
            for (auto q : quad_points()) {
                auto x = point(e, q);
                kq(e[0], e[1], q[0], q[1]) = 1e2;  // permeability function
            }
        }
    }

    void before_step(int /*iter*/, double /*t*/) override {
        using std::swap;
        swap(u, u_prev);
    }

    void step(int /*iter*/, double t) override {
        compute_rhs(t);
        solve(u);
    }

    void compute_rhs(double t) {
        auto& rhs = u;

        zero(rhs);
        executor.for_each(elements(), [&](index_type e) {
            auto U = element_rhs();

            double J = jacobian(e);
            for (auto q : quad_points()) {
                double w = weight(q);
                auto x = point(e, q);

                double mi = 10;
                double k = permeability(e, q);
                value_type u = eval_fun(u_prev, e, q);
                double h = forcing(x, t, u.val);

                for (auto a : dofs_on_element(e)) {
                    auto aa = dof_global_to_local(e, a);
                    value_type v = eval_basis(e, q, a);

                    double val = -k * std::exp(mi * u.val) * grad_dot(u, v) + h * v.val;
                    U(aa[0], aa[1]) += (u.val * v.val + steps.dt * val) * w * J;
                }
            }
            executor.synchronized([&] { update_global_rhs(rhs, U, e); });
        });
    }

    double energy(const vector_type& u) const {
        double E = 0;
        for (auto e : elements()) {
            double J = jacobian(e);
            for (auto q : quad_points()) {
                double w = weight(q);
                value_type a = eval_fun(u, e, q);
                E += a.val * a.val * w * J;
            }
        }
        return E;
    }

    void after_step(int iter, double /*t*/) override {
        if (iter % 10 == 0) {
            std::cout << "Step " << iter << ", energy: " << energy(u) << std::endl;
        }
        if ((iter + 1) % 100 == 0) {
            output.to_file(u, "out_%d.data", iter + 1);
        }
    }

    double permeability(index_type e, index_type q) const { return kq(e[0], e[1], q[0], q[1]); }

    double forcing(point_type x, double /*t*/, double u) const {
        return process.pumping(x[0], x[1]) - process.draining(x[0], x[1], u);
    }
};

}  // namespace ads

#endif  // OIL2D_HPP