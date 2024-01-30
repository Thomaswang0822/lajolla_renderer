#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
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
    Vector3 hl = to_local(frame, h);
    Real ccGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_sq = pow( (1.0 - ccGloss) * 0.1 + ccGloss * 1e-3, 2);
    // std::cout << "alpha_sq: " << alpha_sq << std::endl;
    // Clamp alpha_sq to avoid numerical issues in log(alpha_sq) of D
    alpha_sq = std::clamp(alpha_sq, Real(1e-4), Real(1));
    // Calculate F
    Real R0_eta = Real(0.04); // pow(0.5/2.5, 2)
    Real F = R0_eta + (1 - R0_eta) * pow(1.0 - fabs(dot(h, dir_out)), 5);
    // Calculate D
    Real D = (alpha_sq - 1.0) / ( c_PI * log(alpha_sq) * (1.0 + (alpha_sq - 1.0) * hl.z*hl.z) );
    // Calculate G, we want 0.25^2, see implementation
    Real G = smith_masking_gtr2(to_local(frame, dir_in), 0.5) *
            smith_masking_gtr2(to_local(frame, dir_out), 0.5);

    return make_const_spectrum(0.25) * F * D * G / dot(frame.n, dir_in);
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
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
    
    // variables
    Vector3 h = normalize(dir_in + dir_out);
    Vector3 hl = to_local(frame, h);
    Real n_dot_in = fabs( dot(frame.n, dir_in) );
    Real n_dot_out = fabs( dot(frame.n, dir_out) );
    Real n_dot_h = fabs( dot(frame.n, h) );

    Real ccGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_sq = pow( (1.0 - ccGloss) * 0.1 + ccGloss * 1e-3, 2);
    // Clamp alpha_sq to avoid numerical issues in log(alpha_sq) of D
    alpha_sq = std::clamp(alpha_sq, Real(0.01), Real(1));
    Real D = (alpha_sq - 1.0) / ( c_PI * log(alpha_sq) * (1.0 + (alpha_sq - 1.0) * hl.z*hl.z) );
    // D * nh / (4 hout)
    return D * n_dot_h / (4.0 * fabs(dot(h, dir_out)));
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    // compute alpha = alpha_g
    Real ccGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_sq = pow( (1.0 - ccGloss) * 0.1 + ccGloss * 1e-3, 2);
    // Clamp alpha_sq to avoid numerical issues in log(alpha_sq) of D
    alpha_sq = std::clamp(alpha_sq, Real(0.01), Real(1));
    // get h in world frame
    Vector3 h = to_world(frame, sample_clear_coat(alpha_sq, rnd_param_uv));
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, h) * h);

    return BSDFSampleRecord{
            reflected,
            Real(1.5) /* fixed eta */, Real(0.25) /* fixed roughness */};
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
