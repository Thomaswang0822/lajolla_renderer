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
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real aspect = sqrt(1.0 - 0.9 * aniso);
    Real alphax = max(1e-4, roughness*roughness/aspect);
    Real alphay = max(1e-4, roughness*roughness*aspect);
    // calculate F
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum F = baseColor + (1.0 - baseColor) * pow(1 - fabs(dot(h, dir_out)), 5);
    // calculate D, can't use GTR2() since we have an anisotropic version
    Real D = D_metal(hl, alphax, alphay);
    // calculate G, also anisotropic version
    Real G = smith_masking_aniso(to_local(frame, dir_in), alphax, alphay) *
        smith_masking_aniso(to_local(frame, dir_out), alphax, alphay);

    return Real(0.25) * F * D * G / fabs( dot(frame.n, dir_in) );

}

// modified version of Disney Metal when putting 5 together
Spectrum eval_op::operator()(const DisneyMetal &bsdf, 
        Real specular, Real metallic, Real spTint, Real eta) const {
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
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real aspect = sqrt(1.0 - 0.9 * aniso);
    Real alphax = max(1e-4, roughness*roughness/aspect);
    Real alphay = max(1e-4, roughness*roughness*aspect);

    // calculate F: DIFFERENT
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real lum = luminance(baseColor);
    Spectrum C_tint = lum > 0 ? baseColor / lum : make_const_spectrum(1);
    Spectrum Ks = (1.0 - spTint)  + spTint * C_tint;
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real R0_eta = pow((eta - 1.0) / (eta + 1.0), 2);
    Spectrum C = specular * R0_eta * (1.0 - metallic) * Ks + metallic * baseColor;
    Spectrum F = C + (1.0 - C) * pow(1 - fabs(dot(h, dir_out)), 5);
    // calculate D, can't use GTR2() since we have an anisotropic version
    Real D = D_metal(hl, alphax, alphay);
    // calculate G, also anisotropic version
    Real G = smith_masking_aniso(to_local(frame, dir_in), alphax, alphay) *
        smith_masking_aniso(to_local(frame, dir_out), alphax, alphay);

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
    Real n_dot_in = fabs( dot(frame.n, dir_in) );
    Real n_dot_out = fabs( dot(frame.n, dir_out) );
    Real n_dot_h = fabs( dot(frame.n, h) );
    // if (n_dot_out <= 0 || n_dot_h <= 0) {
    //     return 0;
    // }
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = sqrt(1.0 - 0.9 * aniso);
    Real alphax = max(1e-4, roughness*roughness/aspect);
    Real alphay = max(1e-4, roughness*roughness*aspect);
    
    // restore G and D
    Real G = smith_masking_aniso(to_local(frame, dir_in),  alphax, alphay);
    Real D = D_metal(hl, alphax, alphay);
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
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = sqrt(1.0 - 0.9 * aniso);
    Real alphax = max(1e-4, roughness*roughness/aspect);
    Real alphay = max(1e-4, roughness*roughness*aspect);
    Vector3 local_micro_normal =
        sample_visible_normals_aniso(local_dir_in, alphax, alphay, rnd_param_uv);
    
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
