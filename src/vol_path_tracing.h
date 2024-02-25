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
                update_medium(*vertex_, ray, curr_medium_id);
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

/**
 * @brief Next Event Estimation, monochromatic version used by version 4 and 5
 * 
 * @param scene 
 * @param vertex intersection point and related data
 * @param dir_in 
 * @param curr_md_id 
 * @param bounces 
 * @param rng 
 * @param scatter 
 * @return Spectrum 
 */
inline Spectrum next_event_est_mono(const Scene& scene, PathVertex& vertex, Vector3& dir_in,
        int curr_md_id, int bounces, pcg32_state &rng, bool scatter=true) {
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
        sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
    Vector3 dir_light = normalize(point_on_light.position - vertex.position); // w'
    Vector3 p = vertex.position; // update p; vertex.position const
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
    if (length_squared(T_light) > 1e-6) { // T_light > 0
        // Compute T_light * G * rho * L & pdf_nee
        G = fabs(dot(dir_light, point_on_light.normal)) /
            distance_squared(point_on_light.position, vertex.position);
        pdf_nee = light_pmf(scene, light_id) *
            pdf_point_on_light(light, point_on_light, vertex.position, scene);  // ref point is the original
        L = emission(light, -dir_light, 0.0, point_on_light, scene); // hopefully footprint isn't used here
        /*
            # Multiple importance sampling: it’s also possible
            # that a phase function sampling + multiple exponential sampling
            # will reach the light source.
            # We also need to multiply with G to convert phase function PDF to area measure.
         */
        // version 5 update: distinguish volume and surface, phase function vs BSDF
        if (scatter) {
            assert(curr_md_id >= 0 && curr_md_id < scene.media.size());
            const Medium& md = scene.media[curr_md_id];
            const PhaseFunction& phaseF = get_phase_function(md);
            // include sigma_s in f, as BSDF f doesn't have it
            f = eval(phaseF, dir_in, dir_light) * get_sigma_s(md, vertex.position);

            pdf_phase = pdf_sample_phase(phaseF, dir_in, dir_light) * G * p_trans_dir;
        } else {
            // for bsdf, material id is stored in vertex
            assert(vertex.material_id >= 0 && vertex.material_id < scene.materials.size());
            const Material& mat = scene.materials[vertex.material_id];
            f = eval(mat, dir_in, dir_light, vertex, scene.texture_pool);

            pdf_phase = pdf_sample_bsdf(mat, dir_in, dir_light, vertex, scene.texture_pool) 
                    * G * p_trans_dir;
        }
        contrib = T_light * G * f * L / pdf_nee;

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
                update_medium(*vertex_, ray, curr_medium_id);
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
            radiance += current_path_throughput * next_event_est_mono(scene, vertex, -ray.dir, curr_medium_id, bounces, rng);
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
            // update dir_pdf
            dir_pdf = pdf_sample_phase(phaseF, -ray.dir, next_dir);
            current_path_throughput *= sigma_s * (
                eval(phaseF, -ray.dir, next_dir) /
                dir_pdf
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
    // COPIED FROM VERSION 4
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
    int max_depth = scene.options.max_depth;

    while (1) {
        // std::cout << "x, y, bounces:" << x << "\t" << y << "\t" << bounces << std::ends;
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
                // AND medium id!
                vertex.exterior_medium_id = curr_medium_id;
                vertex.interior_medium_id = curr_medium_id;
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

        if (!scatter && !vertex_) {
            break;
        }

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
                update_medium(*vertex_, ray, curr_medium_id);
                bounces++;
                continue;
            }
        }

        // since both if and else is a "scatter", we should reset outside
        multi_trans_pdf = 1.0;
        nee_p_cache = vertex.position;  // for later loop use

        // New Step 6B: nee sampling; if (scatter) is inside nee()
        /* CANNOT use *vertex_, because 
        1) it can be null 
        2) nee uses scatter point = vertex.position !=(may) vertex_->position */
        radiance += current_path_throughput * next_event_est_mono(scene, vertex, -ray.dir, curr_medium_id, bounces, rng, scatter);

        // Step 6: scatter, update path throughput
        rnd_param.x = next_pcg32_real<Real>(rng);
        rnd_param.y = next_pcg32_real<Real>(rng);
        if (scatter) {
            // Phase function (dir) sampling
            // updated vertex.position, thus update sigma
            assert(curr_medium_id >= 0 && curr_medium_id < scene.media.size());
            const Medium& md = scene.media[curr_medium_id];  // GET NEW MEDIUM!!!
            const PhaseFunction& phaseF = get_phase_function(md);
            sigma_s = get_sigma_s(md, vertex.position);

            next_dir_ = sample_phase_function(phaseF, -ray.dir, rnd_param);
            // what to do if sample_phase failed? Looks like it won't
            if (!next_dir_) {break;}
            next_dir = *next_dir_ ;
            // update dir_pdf
            dir_pdf = pdf_sample_phase(phaseF, -ray.dir, next_dir);
            current_path_throughput *= sigma_s * (
                eval(phaseF, -ray.dir, next_dir) /
                dir_pdf
            );
        } else {
            assert(vertex_);
            // New Step 6C: BSDF sampling
            assert(vertex_->material_id >= 0 && vertex_->material_id < scene.materials.size());
            const Material& mat = scene.materials[vertex_->material_id];

            std::optional<BSDFSampleRecord> bsdf_sample_ = sample_bsdf(mat, -ray.dir, *vertex_, 
                    scene.texture_pool, rnd_param, next_pcg32_real<Real>(rng));
            if (!bsdf_sample_) {break;}
            next_dir = bsdf_sample_->dir_out;
            // update dir_pdf
            dir_pdf = pdf_sample_bsdf(mat, -ray.dir, next_dir, *vertex_, scene.texture_pool);
            current_path_throughput *= (
                eval(mat, -ray.dir, next_dir, *vertex_, scene.texture_pool) /
                dir_pdf
            );
            // hitting a concrete surface should also be a "scatter"
            scatter = true;
            never_scatter = false;
        }
        // and update ray in both cases
        ray = {vertex.position, next_dir, get_intersection_epsilon(scene), infinity<Real>()};
        update_medium(vertex, ray, curr_medium_id);

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

// helper: update all vars related to a Medium (but not majorant)
inline void fill_sigma_data(const Medium &md, const Vector3 &p, Spectrum &maj, 
        Spectrum &s, Spectrum &t, Spectrum &n)
{
    // again, we don't need sigma_a
    s = get_sigma_s(md, p);
    t = get_sigma_a(md, p) + s;
    n = maj - t;
}

/**
 * @brief Next Event Estimation, chromatic version used by final version
 * 
 * @param scene 
 * @param vertex intersection point and related data
 * @param dir_in 
 * @param curr_md_id 
 * @param bounces 
 * @param rng 
 * @param scatter 
 * @return Spectrum 
 */
inline Spectrum next_event_est_chromatic(const Scene& scene, PathVertex& vertex, Vector3& dir_in,
        int curr_md_id, int bounces, pcg32_state &rng, bool scatter=true) {
    // vars
    int shadow_md_id = curr_md_id;
    int shadow_bounces = 0;
    const int max_depth = scene.options.max_depth;
    const int max_null_coll = scene.options.max_null_collisions;
    int iteration, ch012;
    Spectrum sigma_s, sigma_t, sigma_n, majorant;
    Ray shadow_ray;
    std::optional<PathVertex> isect_;
    Real next_t;  // length of ray in current segment
    Real t, dt, acc_t;
    Real real_prob;
    // Compute transmittance to light. Skip through index-matching shapes.
    Spectrum T_light = make_const_spectrum(1.0);
    Spectrum p_trans_dir = make_const_spectrum(1.0); // for multiple importance sampling
    Spectrum p_trans_nee = make_const_spectrum(1.0);
    
    // sample a light, copied from path_tracing()
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light =
        sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
    Vector3 dir_light = normalize(point_on_light.position - vertex.position); // w'
    Vector3 p = vertex.position; // update p; vertex.position const
    Vector3 p_prime = point_on_light.position;

    // Let shadow ray pass thru all index-matching medium (under recursion limit)
    while (1) {
        // shoot and intersect shadow ray
        shadow_ray = {p, dir_light, 
                get_shadow_epsilon(scene), (1 - get_shadow_epsilon(scene)) * distance(p, p_prime)};
        isect_ = intersect(scene, shadow_ray);
        // next_t = distance(p, p_prime);
        next_t = isect_? distance(p, isect_->position) : shadow_ray.tfar;

        // Account for the transmittance to next_t
        if (shadow_md_id >= 0) {
            const Medium& shadow_md = scene.media[shadow_md_id];
            majorant = get_majorant(shadow_md, shadow_ray);
            
            // choose an RGB channel and do sampling
            ch012 = std::clamp(
                static_cast<int>(next_pcg32_real<Real>(rng) * 3),   // next_pcg32_real<int>(rng) may be wrong
                0, 2
            );
            acc_t = 0.0;
            iteration = 0;
            
            while(1) {
                if (majorant[ch012] <= 0) 
                    break;
                if (iteration >= max_null_coll)
                    break;
                t = -log(1.0 - next_pcg32_real<Real>(rng)) / majorant[ch012];
                dt = next_t - acc_t;
                acc_t = min(acc_t + t, next_t);
                // update p and parameters
                p = shadow_ray.org + acc_t * shadow_ray.dir;
                fill_sigma_data(shadow_md, p, majorant, sigma_s, sigma_t, sigma_n);
                if (t < dt) {
                    // didn’t hit the surface, so this is a null-scattering event
                    T_light *= exp(-majorant * t) * sigma_n / max(majorant);
                    p_trans_nee *= exp(-majorant * t) * majorant / max(majorant);
                    real_prob = (sigma_t / majorant)[ch012];
                    p_trans_dir *= exp(-majorant * t) * majorant * (1 - real_prob) / max(majorant);
                    if (max(T_light) <= 0) // optimization for places where sigma_n = 0;
                        break;
                } else {
                    // hit the surface
                    T_light *= exp(-majorant * dt);
                    p_trans_nee *= exp(-majorant * dt);
                    p_trans_dir *= exp(-majorant * dt);
                    break;
                }
                iteration++;
            }
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
    if (max(T_light) > 0) { // T_light > 0
        // Compute T_light * G * rho * L & pdf_nee
        G = fabs(dot(dir_light, point_on_light.normal)) /
            distance_squared(point_on_light.position, vertex.position);
        pdf_nee = light_pmf(scene, light_id) *
            pdf_point_on_light(light, point_on_light, vertex.position, scene) *   // ref point is the original
            average(p_trans_nee);  // ratio-tracking pdf
        L = emission(light, -dir_light, 0.0, point_on_light, scene); // hopefully footprint isn't used here
        /*
            # Multiple importance sampling: it’s also possible
            # that a phase function sampling + multiple exponential sampling
            # will reach the light source.
            # We also need to multiply with G to convert phase function PDF to area measure.
         */
        // version 5 update: distinguish volume and surface, phase function vs BSDF
        if (scatter) {
            assert(curr_md_id >= 0 && curr_md_id < scene.media.size());
            const Medium& md = scene.media[curr_md_id];
            const PhaseFunction& phaseF = get_phase_function(md);
            // include sigma_s in f, as BSDF f doesn't have it
            f = eval(phaseF, dir_in, dir_light) * get_sigma_s(md, vertex.position);

            pdf_phase = pdf_sample_phase(phaseF, dir_in, dir_light) * G;
        } else {
            // for bsdf, material id is stored in vertex
            assert(vertex.material_id >= 0 && vertex.material_id < scene.materials.size());
            const Material& mat = scene.materials[vertex.material_id];
            f = eval(mat, dir_in, dir_light, vertex, scene.texture_pool);

            pdf_phase = pdf_sample_bsdf(mat, dir_in, dir_light, vertex, scene.texture_pool) * G;
        }
        pdf_phase *= average(p_trans_dir);  // NEW
        contrib = T_light * G * f * L / pdf_nee;

        // power heuristics
        w = pdf_nee * pdf_nee / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
        return w * contrib;
    }
    return make_zero_spectrum();
}


// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    #pragma region variable declaration
    // ******* key vars *******
    Spectrum current_path_throughput = make_const_spectrum(1.0);
    Spectrum radiance = make_zero_spectrum();
    bool scatter = false, never_scatter = true;
    // a fake vertex hit in case vertex_ is nullptr
    PathVertex vertex; std::optional<PathVertex> vertex_;
    Spectrum transmittance;  // T term
    // Many PDFs
    Real dir_pdf = 0.0; // the pdf (solid angle measure) of the latest phase function (dir) sampling;
    Vector3 nee_p_cache; // the last position p that can issue a nee
    // pdf for free-flight sampling and nee
    Spectrum trans_dir_pdf, trans_nee_pdf;
    // product PDF of free-flight (transmittance) sampling / nee going through
    //     several index-matching surfaces from the last phase function sampling
    Spectrum multi_trans_dir_pdf = make_const_spectrum(1);
    Spectrum multi_trans_nee_pdf = make_const_spectrum(1);
    Real t_hit, t;
    Real acc_t;  // accumulated t
    Real dt;  // dist to t_hit (hit point)
    Vector3 acc_p;  // o + dir * acc_t
    
    // useful vars
    Real wMIS; // MIS weight
    Real rr_prob = Real(1.0); // prob of not terminating
    Real real_prob;  // sampling real or fake particle
    Spectrum sigma_s, sigma_t, sigma_n, majorant;
    std::optional<Vector3> next_dir_; Vector3 next_dir;
    Vector2 rnd_param; 
    const int max_depth = scene.options.max_depth;
    const int max_null_coll = scene.options.max_null_collisions;
    int curr_medium_id = scene.camera.medium_id;
    int light_id;  // randomly pick a light
    int bounces = 0;  // outer loop count
    int iteration;  // inner loop count
    int ch012;  // chromatic: which RGB channel; 0 or 1 or 2
    #pragma endregion variable declaration

    // trace camera ray and intersect the scene
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);

    while (1) {
        // std::cout << "x, y, bounces:" << x << "\t" << y << "\t" << bounces << std::ends;
        scatter = false;
        vertex_ = intersect(scene, ray);
        vertex = vertex_ ? *vertex_ : PathVertex();
        // reset per-loop value
        transmittance = make_const_spectrum(1.0);
        trans_dir_pdf = make_const_spectrum(1.0);
        trans_nee_pdf = make_const_spectrum(1.0);

        // Step 1: if in a medium, sample t and compute trans_dir_pdf and transmittance
        #pragma region step 1
        if (curr_medium_id >= 0) {
            // if not hit and id > 0, scatter = true; !scatter -> must have a hit
            t_hit = vertex_ ? distance(vertex_->position, ray.org) : infinity<Real>();
            const Medium& md = scene.media[curr_medium_id];
            majorant = get_majorant(md, ray);

            // choose an RGB channel and do sampling
            ch012 = std::clamp(
                static_cast<int>(next_pcg32_real<Real>(rng) * 3),   // next_pcg32_real<int>(rng) may be wrong
                0, 2
            );
            acc_t = Real(0.0);
            iteration = 0;

            while (1) {
                if (majorant[ch012] <= 0)
                    break;
                if (iteration >= max_null_coll)
                    break;
                t = -log(1.0 - next_pcg32_real<Real>(rng)) / majorant[ch012];
                dt = t_hit - acc_t;
                // Update accumulated distance
                acc_t = min(acc_t + t, t_hit);

                if (t < dt) {   // haven't reach surface
                    // fill in data
                    acc_p = ray.org + ray.dir * acc_t;
                    fill_sigma_data(md, acc_p, majorant, sigma_s, sigma_t, sigma_n);
                    // sample from real/fake particle events
                    real_prob = (sigma_t / majorant)[ch012];
                    if (next_pcg32_real<Real>(rng) < real_prob) {
                        // real particle
                        scatter = true;
                        never_scatter = false;
                        transmittance *= exp(-majorant * t) / max(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * majorant * real_prob / max(majorant);
                        // just like before (t < t_hit), update potentially null Vertex
                        vertex.position = ray.org + acc_t * ray.dir;
                        vertex.exterior_medium_id = curr_medium_id;
                        vertex.interior_medium_id = curr_medium_id;
                        // don’t need to account for trans_nee_pdf since we scatter
                        break;
                    } else {
                        // fake particle
                        transmittance *= exp(-majorant * t) * sigma_n / max(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * majorant * (1 - real_prob) / max(majorant);
                        trans_nee_pdf *= exp(-majorant * t) * majorant / max(majorant);
                    }
                } else {  // reached a surface
                    transmittance *= exp(-majorant * dt);
                    trans_dir_pdf *= exp(-majorant * dt);
                    trans_nee_pdf *= exp(-majorant * dt);
                    break;
                }
                iteration++;
            }  // end inner while
        }
        #pragma endregion step 1

        // Step 2: update throughtput AND pdf product
        multi_trans_dir_pdf *= trans_dir_pdf;
        multi_trans_nee_pdf *= trans_nee_pdf;
        // average PDFs over the RGB channels.
        current_path_throughput *= (transmittance / average(trans_dir_pdf));

        if (!scatter && !vertex_) {
            break;
        }

        // Step 3: no scatter (reach a surface), include possible emission
        #pragma region step 3
        if (!scatter && vertex_ && is_light(scene.shapes[vertex_->shape_id])) { // t > t_hit
            if (never_scatter) {
                // This is the only way we can see the light source, so
                // we don’t need multiple importance sampling.
                radiance += current_path_throughput * emission(vertex, -ray.dir, scene);
            } else {
                /* Account for next event estimation:
                this isect point can be found by both nee (light sampling) and dir sampling,
                we consider contrib by dir sampling under MIS here */
                assert(vertex_);
                // grab Light, don't sample (isect point is the point of interest)
                light_id = get_area_light_id(scene.shapes[vertex_->shape_id]);
                assert(light_id >= 0 && light_id < scene.lights.size());
                const Light& light = scene.lights[light_id];
                PointAndNormal point_on_light = {vertex_->position, vertex_->geometric_normal};
                // compute G and dir_pdf_temp: phase function sampling + transmittance sampling
                Vector3 dir_light = normalize(vertex_->position - nee_p_cache);
                Real G = fabs(dot(dir_light, point_on_light.normal)) /
                        distance_squared(point_on_light.position, nee_p_cache);
                // find nee_pdf_temp by nee() issued by nee_p_cache several bounces ago
                // NEW: also account for ratio tracking pdf
                Real nee_pdf_temp = light_pmf(scene, light_id) * 
                        pdf_point_on_light(light, point_on_light, nee_p_cache, scene) *
                        average(multi_trans_nee_pdf);  // ratio tracking
                // and dir_pdf_temp to update radiance with MIS weight
                Real dir_pdf_temp = dir_pdf * average(multi_trans_dir_pdf) * G;
                wMIS = dir_pdf_temp * dir_pdf_temp / (dir_pdf_temp * dir_pdf_temp + nee_pdf_temp * nee_pdf_temp);
                radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * wMIS;
            }
        }
        #pragma endregion step 3
        

        // Step 4: terminate if reach max_depth if not Russian-Roulette
        if (bounces == scene.options.max_depth - 1 && scene.options.max_depth != -1)
            break;

        // Step 5: no scatter and hit -> if index-matching interface, skip through it
        if (!scatter && vertex_) {
            if (vertex_->material_id == -1) {
                // update ray, same dir, before update medium
                ray = {vertex.position, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
                update_medium(*vertex_, ray, curr_medium_id);
                bounces++;
                continue;
            }
        }

        // since both if and else is a "scatter", we should reset outside
        multi_trans_dir_pdf = make_const_spectrum(1.0);
        multi_trans_nee_pdf = make_const_spectrum(1.0);
        nee_p_cache = vertex.position;  // for later loop use

        // next event estimation
        /* CANNOT use *vertex_, because 
        1) it can be null 
        2) nee uses scatter point = vertex.position !=(may) vertex_->position */
        radiance += current_path_throughput * next_event_est_chromatic(scene, vertex, -ray.dir, curr_medium_id, bounces, rng, scatter);

        // Step 6: scatter, update path throughput
        #pragma region step 6
        rnd_param.x = next_pcg32_real<Real>(rng);
        rnd_param.y = next_pcg32_real<Real>(rng);
        if (scatter) {
            // Phase function (dir) sampling
            // updated vertex.position, thus update sigma
            assert(curr_medium_id >= 0 && curr_medium_id < scene.media.size());
            const Medium& md = scene.media[curr_medium_id];  // GET NEW MEDIUM!!!
            const PhaseFunction& phaseF = get_phase_function(md);
            sigma_s = get_sigma_s(md, vertex.position);

            next_dir_ = sample_phase_function(phaseF, -ray.dir, rnd_param);
            // what to do if sample_phase failed? Looks like it won't
            if (!next_dir_) {break;}
            next_dir = *next_dir_ ;
            // update dir_pdf
            dir_pdf = pdf_sample_phase(phaseF, -ray.dir, next_dir);
            current_path_throughput *= sigma_s * (
                eval(phaseF, -ray.dir, next_dir) /
                dir_pdf
            );
        } else {
            assert(vertex_);
            // BSDF sampling
            assert(vertex_->material_id >= 0 && vertex_->material_id < scene.materials.size());
            const Material& mat = scene.materials[vertex_->material_id];

            std::optional<BSDFSampleRecord> bsdf_sample_ = sample_bsdf(mat, -ray.dir, *vertex_, 
                    scene.texture_pool, rnd_param, next_pcg32_real<Real>(rng));
            if (!bsdf_sample_) {break;}
            next_dir = bsdf_sample_->dir_out;
            // update dir_pdf
            dir_pdf = pdf_sample_bsdf(mat, -ray.dir, next_dir, *vertex_, scene.texture_pool);
            current_path_throughput *= (
                eval(mat, -ray.dir, next_dir, *vertex_, scene.texture_pool) /
                dir_pdf
            );
            // hitting a concrete surface should also be a "scatter"
            scatter = true;
            never_scatter = false;
        }
        // and update ray in both cases
        ray = {vertex.position, next_dir, get_intersection_epsilon(scene), infinity<Real>()};
        update_medium(vertex, ray, curr_medium_id);
        #pragma endregion step 6

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
