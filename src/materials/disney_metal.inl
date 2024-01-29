#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
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
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // calculate F
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum F = baseColor + (1.0 - baseColor) * pow(1 - fabs(dot(h, dir_out)), 5);
    // calculate D, can't use GTR2() since we have an anisotropic version
    Real D = D_metal(hl, aniso, roughness);
    // calculate G, same as rough_plastic
    Real G = smith_masking_gtr2(to_local(frame, dir_in), roughness) *
        smith_masking_gtr2(to_local(frame, dir_out), roughness);

    return Real(0.25) * F * D * G / dot(frame.n, dir_in);

}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
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
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, h);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return 0;
    }
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // restore G and D
    Real G = smith_masking_gtr2(to_local(frame, dir_in), roughness);
    Real D = D_metal(hl, aniso, roughness);
    // (4 * cos_theta_v) is the Jacobian of the reflection
    return G * D / (4.0 * n_dot_in);
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Convert the incoming direction to local coordinates
    Vector3 local_dir_in = to_local(frame, dir_in);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real alpha = roughness * roughness;
    Vector3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha, rnd_param_uv);
    
    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, roughness /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
