#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneySheen &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // variables
    Vector3 h = normalize(dir_in + dir_out);
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real lum = luminance(baseColor);
    Real sheenTint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    // C
    Spectrum C_tint = lum > 0 ? baseColor / lum : make_const_spectrum(1);
    Spectrum C = (1.0 - sheenTint) + sheenTint * C_tint;

    return C * pow(1.0 - fabs(dot(h, dir_out)), 5) * fabs(dot(frame.n, dir_out));
}

Real pdf_sample_bsdf_op::operator()(const DisneySheen &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // We importance sample the cosine hemisphere domain.
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneySheen &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // simply do cos hemisphere sampling
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, Real(1) /* roughness */};
}

TextureSpectrum get_texture_op::operator()(const DisneySheen &bsdf) const {
    return bsdf.base_color;
}
