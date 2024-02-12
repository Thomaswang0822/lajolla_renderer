#pragma once

// Updates the current medium the ray lies in
inline void update_medium(const PathVertex& vertex, const Ray& ray, int& m_id) {
    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
        if (dot(ray.dir, vertex.geometric_normal) > 0) {
            m_id = vertex.exterior_medium_id;
        } else {
            m_id = vertex.interior_medium_id;
        }
    }
}

// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // trace camera ray and intersect the scene
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);
    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        return make_zero_spectrum();
    }

    PathVertex vertex = *vertex_;
    // compute transmittance = exp(− σa * t)
    Real t = distance(vertex.position, ray.org);  // seems like ray.tfar is not updated
    // assert(vertex.exterior_medium_id == vertex.interior_medium_id);
    Medium md = scene.media[scene.camera.medium_id];
    Spectrum sigma_a = get_sigma_a(md, ray.org);
    Spectrum transmittance = exp(-t * sigma_a);

    Spectrum Le = make_zero_spectrum();
    if (is_light(scene.shapes[vertex.shape_id])) {
        Le = emission(vertex, -ray.dir, scene);
    }
    // std::cout << transmittance << Le << std::endl;
    return transmittance * Le;
}

// The second simplest volumetric renderer: 
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    
    // trace camera ray and intersect the scene
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);
    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    // other vars
    Real t_hit = vertex_ ? distance(vertex_->position, ray.org) : infinity<Real>();
    Medium md = scene.media[scene.camera.medium_id];
    Spectrum sigma_s = get_sigma_s(md, ray.org);
    Spectrum sigma_t = get_sigma_a(md, ray.org) + sigma_s;
    PhaseFunction phaseF = get_phase_function(md);
    //  importance sample the transmittance exp(−σ_t * t)
    Real t = -log(1.0 - next_pcg32_real<Real>(rng)) / sigma_t.x;
    // and we create a fake vertex hit based on sampled t
    PathVertex vertex;
    vertex.position = ray.org + t * ray.dir;

    // We need to account for the probability of that t >= t_hit
    if (t < t_hit) {
        Real trans_pdf = exp(-sigma_t.x * t) * sigma_t.x;
        Spectrum transmittance = exp(-sigma_t * t);
        
        // The 2 end results of Monte Carlo sampling (on light)
        Spectrum L_s1 = make_zero_spectrum(); 
        Real L_s1_pdf = 1.0;
        // sample a light, copied from path_tracing()
        // We do this by first picking a light source, then pick a point on it.
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
        // no MIS; L_s1_estimate in eq (7): f (phase function) * L * transmittance * G (Jacobian)
        Vector3 dir_light = normalize(point_on_light.position - vertex.position); // w'
        Ray shadow_ray{vertex.position, dir_light,
                get_shadow_epsilon(scene),
                (1 - get_shadow_epsilon(scene)) *
                    distance(point_on_light.position, vertex.position)};\
        if (occluded(scene, shadow_ray)) {
            // G will be 0 and make everything 0
            return make_zero_spectrum();
        } 

        Real G = fabs(dot(dir_light, point_on_light.normal)) /
                distance_squared(point_on_light.position, vertex.position);
        // compute light pdf
        L_s1_pdf = light_pmf(scene, light_id) *
                pdf_point_on_light(light, point_on_light, vertex.position, scene);
        // continue only when no occlusion + possibl light path
        if (G > 0 && L_s1_pdf > 0) {
            Vector3 dir_view = -ray.dir;
            Spectrum f = eval(phaseF, dir_view, dir_light);
            Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);
            Spectrum T = exp(-sigma_t * distance(point_on_light.position, vertex.position));
            L_s1 = f * L * T * G;
        }
        return (transmittance / trans_pdf) * sigma_s * (L_s1 / L_s1_pdf);
    } else {
        // hit a surface, account for surface emission
        Real trans_pdf = exp(-sigma_t.x * t_hit);
        Spectrum transmittance = exp(-sigma_t * t_hit);
        Spectrum Le = make_zero_spectrum();
        if (is_light(scene.shapes[vertex_->shape_id])) {
            Le = emission(*vertex_, -ray.dir, scene);
        }
        return (transmittance / trans_pdf) * Le;
    }
}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // trace camera ray and intersect the scene
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);
    int curr_medium_id = scene.camera.medium_id;

    // other vars
    Spectrum current_path_throughput = make_const_spectrum(1.0);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    bool scatter;
    // a fake vertex hit in case vertex_ is nullptr
    PathVertex vertex; std::optional<PathVertex> vertex_;
    Spectrum transmittance;
    Real trans_pdf;
    Real t_hit, t;
    Spectrum sigma_s, sigma_t;
    PhaseFunction phaseF;
    std::optional<Vector3> next_dir_; Vector3 next_dir;
    Vector2 rnd_param; Real rr_prob = Real(1.0); // prob of not terminating

    while (1) {
        scatter = false;
        vertex_ = intersect(scene, ray, ray_diff);
        if (vertex_) {
            vertex = *vertex_;
        } else {
            vertex = PathVertex();
        }
        // isect might not intersect a surface, but we might be in a volume
        transmittance = make_const_spectrum(1.0);
        trans_pdf = Real(1.0);
        // Step 1: if in a medium, sample t and compute trans_pdf and transmittance
        if (curr_medium_id >= 0) {
            // if not hit and id > 0, scatter = true; !scatter -> must have a hit
            t_hit = vertex_ ? distance(vertex_->position, ray.org) : infinity<Real>();
            const Medium& md = scene.media[scene.camera.medium_id];
            sigma_s = get_sigma_s(md, ray.org);
            sigma_t = get_sigma_a(md, ray.org) + sigma_s;
            phaseF = get_phase_function(md);
            //  importance sample the transmittance exp(−σ_t * t)
            t = -log(1.0 - next_pcg32_real<Real>(rng)) / sigma_t.x;
            // compute transmittance and trans_pdf
            if (t < t_hit) {
                scatter = true;
                // when vertex is fake, update position: o + dir * t_hit -> o + dir * t
                vertex.position = ray.org + t * ray.dir;
                // but we update sigma s and t later
                trans_pdf = exp(-sigma_t.x * t) * sigma_t.x;
                transmittance = exp(-sigma_t * t);
            } else {
                trans_pdf = exp(-sigma_t.x * t_hit);
                transmittance = exp(-sigma_t * t_hit);
            }
        }

        // Step 2: update throughtput
        current_path_throughput *= (transmittance / trans_pdf);

        // Step 3: no scatter (reach a surface), include possible emission
        if (!scatter && vertex_ && is_light(scene.shapes[vertex_->shape_id])) { // t > t_hit
            radiance += current_path_throughput * emission(vertex, -ray.dir, scene);
        }

        // Step 4: terminate if reach max_depth if not Russian-Roulette
        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
            break;

        // Step 5: no scatter and hit -> if index-matching interface, skip through it
        if (!scatter && vertex_) {
            if (vertex_->material_id == -1) {
                // update ray, same dir, before update medium
                // std::cout << ray.org << ray.dir << ray.tnear << ray.tfar;
                // ray = {vertex.position, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
                // std::cout << ray.org << ray.dir << ray.tnear << ray.tfar;
                update_medium(vertex, ray, curr_medium_id);
                bounces++;
                continue;
            }
        }

        // Step 6: scatter, update path throughput
        if (scatter) {
            // updated vertex.position, thus update sigma
            const Medium& md = scene.media[scene.camera.medium_id];
            sigma_s = get_sigma_s(md, vertex.position);
            sigma_t = get_sigma_a(md, vertex.position) + sigma_s;

            rnd_param.x = next_pcg32_real<Real>(rng);
            rnd_param.y = next_pcg32_real<Real>(rng);
            next_dir_ = sample_phase_function(phaseF, -ray.dir, rnd_param);
            // what to do if sample_phase failed? Looks like it won't
            if (!next_dir_) {break;}
            next_dir = *next_dir_ ;
            current_path_throughput *= sigma_s * (
                eval(phaseF, -ray.dir, next_dir) /
                pdf_sample_phase(phaseF, -ray.dir, next_dir)
            );
            // update ray
            ray = {vertex.position, next_dir, get_intersection_epsilon(scene), infinity<Real>()};
        } else {
            // Hit a surface -- don’t need to deal with this yet
            break;
        }

        // Step 7: Russian-Roulette
        rr_prob = Real(1.0);
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max(current_path_throughput), 0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            } else {
                current_path_throughput /= rr_prob;
            }
        }
        
        bounces++;
    }

    return radiance;
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}
