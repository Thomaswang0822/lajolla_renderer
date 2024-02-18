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
    std::optional<Vector3> next_dir_; Vector3 next_dir;
    Vector2 rnd_param; Real rr_prob = Real(1.0); // prob of not terminating

    while (1) {
        scatter = false;
        vertex_ = intersect(scene, ray, ray_diff);
        vertex = vertex_ ? *vertex_ : PathVertex();
        // isect might not intersect a surface, but we might be in a volume
        transmittance = make_const_spectrum(1.0);
        trans_pdf = Real(1.0);
        // Step 1: if in a medium, sample t and compute trans_pdf and transmittance
        if (curr_medium_id >= 0) {
            // if not hit and id > 0, scatter = true; !scatter -> must have a hit
            t_hit = vertex_ ? distance(vertex_->position, ray.org) : infinity<Real>();
            const Medium& md = scene.media[curr_medium_id];
            sigma_s = get_sigma_s(md, ray.org);
            sigma_t = get_sigma_a(md, ray.org) + sigma_s;
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
                ray = {vertex.position, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
                update_medium(vertex, ray, curr_medium_id);
                bounces++;
                continue;
            }
        }

        // Step 6: scatter, update path throughput
        if (scatter) {
            // updated vertex.position, thus update sigma
            const Medium& md = scene.media[curr_medium_id];  // GET NEW MEDIUM!!!
            const PhaseFunction& phaseF = get_phase_function(md);
            sigma_s = get_sigma_s(md, vertex.position);

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
            // Hit a surface -- no surface lighting yet
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

inline Spectrum next_event_est(const Scene& scene, Vector3& pos, Vector3& dir_in,
        int curr_md_id, int bounces, pcg32_state &rng) {
    // vars
    Real next_t;  // length of ray in current segment
    std::optional<PathVertex> isect_;
    Ray shadow_ray;
    int max_depth = scene.options.max_depth;
    Spectrum sigma_s, sigma_t;

    // sample a light, copied from path_tracing()
    // We do this by first picking a light source, then pick a point on it.
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light =
        sample_point_on_light(light, pos, light_uv, shape_w, scene);
    Vector3 dir_light = normalize(point_on_light.position - pos); // w'
    Vector3 p = pos; // and from now on we should no longer use vertex.position?
    Vector3 p_prime = point_on_light.position;

    // Compute transmittance to light. Skip through index-matching shapes.
    Spectrum T_light = make_const_spectrum(1.0);
    int shadow_md_id = curr_md_id;
    int shadow_bounces = 0;
    Real p_trans_dir = 1.0; // for multiple importance sampling
    // Let shadow ray pass thru all index-matching medium (under recursion limit)
    while (1) {
        // shoot and intersect shadow ray
        shadow_ray = {p, dir_light, 
                get_shadow_epsilon(scene), (1 - get_shadow_epsilon(scene)) * distance(p, p_prime)};
        isect_ = intersect(scene, shadow_ray);
        // next_t = distance(p, p_prime);
        next_t = shadow_ray.tfar;
        // update
        if (isect_) {
            next_t = distance(p, isect_->position);
        }

        // Account for the transmittance to next_t
        if (shadow_md_id >= 0) {
            // shadow_md_id will be update repetitively
            const Medium& shadow_md = scene.media[shadow_md_id];
            sigma_s = get_sigma_s(shadow_md, p);
            sigma_t = get_sigma_a(shadow_md, p) + sigma_s;
            T_light *= exp(-sigma_t * next_t);
            p_trans_dir *= exp(-sigma_t.x * next_t);
        }

        if (!isect_) {
            // Nothing is blocking, we’re done
            // and we can safely access member if isect_ later
            break;
        } else {
            // Something is blocking: is it an opaque surface?
            if (isect_->material_id >= 0){
                // we’re blocked
                return make_zero_spectrum();
            }
            // otherwise, path thru index-matching surface
            shadow_bounces++;
            if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                return make_zero_spectrum();
            }

            update_medium(*isect_, shadow_ray, shadow_md_id);
            // move starting point of next shadow ray to isect point
            p += next_t * dir_light;
        }
    }

    // vars for MIS
    Real G, w;
    Real pdf_nee, pdf_phase;
    Spectrum f, L, contrib;
    const Medium& md = scene.media[curr_md_id];
    const PhaseFunction& phaseF = get_phase_function(md);
    if (length_squared(T_light) > 1e-6) { // T_light > 0
        // Compute T_light * G * rho * L & pdf_nee
        G = fabs(dot(dir_light, point_on_light.normal)) /
            distance_squared(point_on_light.position, pos);
        pdf_nee = light_pmf(scene, light_id) *
            pdf_point_on_light(light, point_on_light, pos, scene);  // ref point is the original
        L = emission(light, -dir_light, 0.0, point_on_light, scene); // hopefully footprint isn't used here
        f = eval(phaseF, dir_in, dir_light);
        sigma_s = get_sigma_s(md, pos);
        contrib = T_light * G * f * L * sigma_s / pdf_nee;
        /*
        # Multiple importance sampling: it’s also possible
        # that a phase function sampling + multiple exponential sampling
        # will reach the light source.
        # We also need to multiply with G to convert phase function PDF to area measure.
        */
        pdf_phase = pdf_sample_phase(phaseF, dir_in, dir_light) * G * p_trans_dir;

        // power heuristics
        w = pdf_nee * pdf_nee / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
        return w * contrib;
    }
    return make_zero_spectrum();
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // COPIED FROM VERSION 3
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
    std::optional<Vector3> next_dir_; Vector3 next_dir;
    Vector2 rnd_param; Real rr_prob = Real(1.0); // prob of not terminating
    // ******* New Cache Vars *******
    Real dir_pdf = 0.0; // the pdf of the latest phase function (dir) sampling;
    Real nee_pdf;
    // product PDF of transmittance sampling going through several index-matching surfaces from the last phase function sampling
    Real multi_trans_pdf = 1.0;
    Real wMIS; // MIS weight
    Vector3 nee_p_cache; // the last position p that can issue a nee
    bool never_scatter = true;
    int light_id;

    while (1) {
        scatter = false;
        vertex_ = intersect(scene, ray, ray_diff);
        vertex = vertex_ ? *vertex_ : PathVertex();
        // isect might not intersect a surface, but we might be in a volume
        transmittance = make_const_spectrum(1.0);
        trans_pdf = Real(1.0);
        // Step 1: if in a medium, sample t and compute trans_pdf and transmittance
        if (curr_medium_id >= 0) {
            // if not hit and id > 0, scatter = true; !scatter -> must have a hit
            t_hit = vertex_ ? distance(vertex_->position, ray.org) : infinity<Real>();
            const Medium& md = scene.media[curr_medium_id];
            sigma_s = get_sigma_s(md, ray.org);
            sigma_t = get_sigma_a(md, ray.org) + sigma_s;
            //  importance sample the transmittance exp(−σ_t * t)
            t = -log(1.0 - next_pcg32_real<Real>(rng)) / sigma_t.x;
            // compute transmittance and trans_pdf
            if (t < t_hit) {
                scatter = true;
                never_scatter = false;
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

        // Step 2: update throughtput AND pdf product
        multi_trans_pdf *= trans_pdf;
        current_path_throughput *= (transmittance / trans_pdf);

        // Step 3: no scatter (reach a surface), include possible emission
        if (!scatter && vertex_ && is_light(scene.shapes[vertex_->shape_id])) { // t > t_hit
            if (never_scatter) {
                // This is the only way we can see the light source, so
                // we don’t need multiple importance sampling.
                radiance += current_path_throughput * emission(vertex, -ray.dir, scene);
            } else {
                assert(vertex_);
                /* New Step 3A: Account for next event estimation:
                this isect point can be found by both nee (light sampling) and dir sampling,
                we consider contribu by dir sampling under MIS here */
                // grab Light, don't sample (isect point is the point of interest)
                light_id = get_area_light_id(scene.shapes[vertex_->shape_id]);
                assert(light_id >= 0 && light_id < scene.lights.size());
                const Light& light = scene.lights[light_id];
                PointAndNormal point_on_light = {vertex_->position, vertex_->geometric_normal};
                // update nee_pdf by nee() issued by nee_p_cache several bounces ago
                nee_pdf = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, nee_p_cache, scene);
                // compute G and dir_pdf_temp: phase function sampling + transmittance sampling
                Vector3 dir_light = normalize(vertex_->position - nee_p_cache);
                Real G = fabs(dot(dir_light, point_on_light.normal)) /
                        distance_squared(point_on_light.position, nee_p_cache);
                Real dir_pdf_temp = dir_pdf * multi_trans_pdf * G;
                // update radiance with MIS weight
                wMIS = dir_pdf_temp * dir_pdf_temp / (dir_pdf_temp * dir_pdf_temp + nee_pdf * nee_pdf);
                radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * wMIS;
            }
        }
        

        // Step 4: terminate if reach max_depth if not Russian-Roulette
        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
            break;

        // Step 5: no scatter and hit -> if index-matching interface, skip through it
        if (!scatter && vertex_) {
            if (vertex_->material_id == -1) {
                // update ray, same dir, before update medium
                ray = {vertex.position, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
                update_medium(vertex, ray, curr_medium_id);
                bounces++;
                continue;
            }
        }
        // no (!scatter && !vertex_) case: if no hit -> t < t_hit = inf -> scatter = true

        // Step 6: scatter, update path throughput
        if (scatter) {
            // reset
            multi_trans_pdf = 1.0;
            nee_p_cache = vertex.position;  // for later loop use
            // New Step 6B: nee sampling
            radiance += current_path_throughput * next_event_est(scene, vertex.position, -ray.dir, curr_medium_id, bounces, rng);
            // Phase function (dir) sampling
            // updated vertex.position, thus update sigma
            const Medium& md = scene.media[curr_medium_id];  // GET NEW MEDIUM!!!
            const PhaseFunction& phaseF = get_phase_function(md);
            sigma_s = get_sigma_s(md, vertex.position);

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
            // Hit a surface -- no surface lighting yet
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
