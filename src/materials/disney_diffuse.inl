inline Spectrum f_baseDiffuse(const DisneyDiffuse &bsdf, const PathVertex &vertex, const TexturePool &texture_pool,
            const Vector3 &dir_in, const Vector3 &dir_out, Vector3 normal) {
    // variables
    Vector3 h = normalize(dir_in + dir_out);
    Real n_dot_out = fabs(dot(normal, dir_out));
    Real n_dot_in = fabs(dot(normal, dir_in));
    Real roughness = eval(
            bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    // intermediate values
    Real F_D90 = Real(0.5) + 2.0 * roughness * pow(dot(h, dir_out), 2);
    Real F_Din = Real(1.0) + (F_D90 - 1.0) * pow(1.0 - n_dot_in, 5);
    Real F_Dout = Real(1.0) + (F_D90 - 1.0) * pow(1.0 - n_dot_out, 5);

    return baseColor / c_PI * F_Din * F_Dout * n_dot_out;
}

inline Spectrum f_subsurface(const DisneyDiffuse &bsdf, const PathVertex &vertex, const TexturePool &texture_pool,
            const Vector3 &dir_in, const Vector3 &dir_out, Vector3 normal) {
    // variables
    Vector3 h = normalize(dir_in + dir_out);
    Real n_dot_out = fabs(dot(normal, dir_out));
    Real n_dot_in = fabs(dot(normal, dir_in));
    Real roughness = eval(
            bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    // intermediate values
    Real F_SS90 = roughness * pow(dot(h, dir_out), 2);
    Real F_SSin = Real(1.0) + (F_SS90 - 1.0) * pow(1.0 - n_dot_in, 5);
    Real F_SSout = Real(1.0) + (F_SS90 - 1.0) * pow(1.0 - n_dot_out, 5);

    return Real(1.25) * baseColor / c_PI * n_dot_out * (
        F_SSin * F_SSout * (1.0 / (n_dot_in + n_dot_out) - 0.5) + 0.5
    );
}

Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
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

    // Homework 1: implement this!
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    return (1.0 - subsurface) * f_baseDiffuse(bsdf, vertex, texture_pool, dir_in, dir_out, frame.n) +
            subsurface * f_subsurface(bsdf, vertex, texture_pool, dir_in, dir_out, frame.n);
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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
    
    // same cos-weighted sampling, same formula
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(
            bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    // Disney Diffuse also use cos-weighted heimisphere sampling
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, roughness};
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
