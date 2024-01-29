#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    
    // extract const
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real spTrans = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real spTint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real ccGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    // if ray is inside the object, keep only glass lobe
    bool inside = dot(dir_in, vertex.geometric_normal) <= 0;

    // reconstruct each component
    DisneyDiffuse diffuse = {bsdf.base_color, bsdf.roughness, bsdf.subsurface};
    DisneyMetal metal = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic};
    DisneyClearcoat clearcoat_struct = {bsdf.clearcoat_gloss};
    DisneyGlass glass = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta};
    DisneySheen sheen_struct = {bsdf.base_color, bsdf.sheen_tint};

    // init the eval_op
    eval_op evalBSDF = {dir_in, dir_out, vertex, texture_pool, dir};

    if (inside) {
        return (1.0 - metallic) * spTrans * evalBSDF(glass);
    }
    else {
        // If we are going into the surface, then we use normal eta
        // (internal/external), otherwise we use external/internal.
        Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        assert(eta > 0 && "Disney eval");

        // compute BSDF of each component
        Spectrum f_diffuse = evalBSDF(diffuse);
        Spectrum f_metal = evalBSDF(metal, specular, metallic, spTint, eta);  // metal has modified version
        Spectrum f_cc = evalBSDF(clearcoat_struct);
        Spectrum f_glass = evalBSDF(glass);
        Spectrum f_sheen = evalBSDF(sheen_struct);

        // blend things together
        Spectrum f = (1.0 - spTrans) * (1.0 - metallic) * f_diffuse +  // diffuse
            (1.0 - metallic) * sheen * f_sheen +                            // sheen
            (1.0 - spTrans * (1.0 - metallic)) * f_metal +// metal
            Real(0.25) * clearcoat * f_cc +                              // clearcoat
            (1.0 - metallic) * spTrans * f_glass; 

        return f;
    }
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    
    // extract const
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real spTrans = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real spTint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real ccGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    // if ray is inside the object, keep only glass lobe
    bool inside = dot(dir_in, vertex.geometric_normal) <= 0;

    // create pdf_sample_bsdf_op
    pdf_sample_bsdf_op pdf = {dir_in, dir_out, vertex, texture_pool, dir};

    // reconstruct each component
    DisneyDiffuse diffuse = {bsdf.base_color, bsdf.roughness, bsdf.subsurface};
    DisneyMetal metal = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic};
    DisneyClearcoat clearcoat_struct = {bsdf.clearcoat_gloss};
    DisneyGlass glass = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta};

    // 4 weights
    Real diffuseW = (1.0 - metallic) * (1.0 - spTrans);
    Real metalW = 1.0 - spTrans * (1.0 - metallic);
    Real glassW = (1.0 - metallic) * spTrans;
    Real ccW = 0.25 * clearcoat;
    // normalize to [0,1]
    Real totalW = diffuseW + metalW + glassW + ccW;
    diffuseW /= totalW;
    metalW /= totalW;
    glassW /= totalW;
    ccW /= totalW;

    // if ray inside obj return glass lobe
    if (inside) {
        return pdf(glass);
    }
    // else 
    return pdf(diffuse) * diffuseW + pdf(metal) * metalW + pdf(glass) * glassW + pdf(clearcoat_struct) * ccW;
}

// Ignore sheen component
std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    
    // extract const
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real spTrans = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real spTint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aniso = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real ccGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    // if ray is inside the object, keep only glass lobe
    bool inside = dot(dir_in, vertex.geometric_normal) <= 0;

    // 4 weights
    Real diffuseW = (1.0 - metallic) * (1.0 - spTrans);
    Real metalW = 1.0 - spTrans * (1.0 - metallic);
    Real glassW = (1.0 - metallic) * spTrans;
    Real ccW = 0.25 * clearcoat;

    // create sample_bsdf_op
    sample_bsdf_op sampler = {dir_in, vertex, texture_pool, rnd_param_uv, rnd_param_w, dir};

    // reconstruct each component
    DisneyDiffuse diffuse = {bsdf.base_color, bsdf.roughness, bsdf.subsurface};
    DisneyMetal metal = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic};
    DisneyClearcoat clearcoat_struct = {bsdf.clearcoat_gloss};
    DisneyGlass glass = {bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta};

    if (inside) {
        return sampler(glass);
    }
    // else
    // there should be 4 bounds: a, a+b, a+b+c, a+b+c+d = 1, normalized to [0,1]
    Real totalW = diffuseW + metalW + glassW + ccW;
    Real b1 = diffuseW / totalW;
    Real b2 = b1 + metalW / totalW;
    Real b3 = b2 + glassW / totalW;

    // use rnd_param_w to determine which lobe to sample
    if (rnd_param_w < b1) {
        return sampler(diffuse);
    } else if (rnd_param_w >= b1 && rnd_param_w < b2) {
        return sampler(metal);
    } else if (rnd_param_w >= b2 && rnd_param_w < b3) {
        return sampler(glass);
    } else {
        return sampler(clearcoat_struct);
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
