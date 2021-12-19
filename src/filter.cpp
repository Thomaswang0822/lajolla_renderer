#include "filter.h"

struct sample_op {
    Vector2 operator()(const Box &filter) const;
    Vector2 operator()(const Tent &filter) const;
    Vector2 operator()(const Gaussian &filter) const;

    const Vector2 &rnd_param;
};

// Implementations of the individual filters.
#include "filter/box.inl"
#include "filter/tent.inl"
#include "filter/gaussian.inl"

Vector2 sample(const Filter &filter, const Vector2 &rnd_param) {
    return std::visit(sample_op{rnd_param}, filter);
}
