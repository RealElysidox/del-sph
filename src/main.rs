use std::{fs::File, io::{BufWriter, Write}};

use del_sph::srchgrid;
use nalgebra;
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)]
struct s_particle {
    x: nalgebra::Vector3<f32>,
    v: nalgebra::Vector3<f32>,
    rho: f32,
    p: f32,
    f: nalgebra::Vector3<f32>,
}

fn cubic_square_distance(ps0: &s_particle, ps1: &s_particle, rad_cutoff: f32) -> f32 {
    let dr = ps0.x - ps1.x;
    if rad_cutoff < dr.norm() {
        0.0
    } else {
        let c = rad_cutoff.powf(2.0) - dr.norm_squared();
        c.powf(3.0)
    }
}

fn density_pressure(
    ps: &mut Vec<s_particle>,
    rad_cutoff: f32,
    p_mass: f32,
    rest_density: f32,
    int_stiff: f32,
) {
    let poly6_kern = 315.0 / (64.0 * std::f32::consts::PI * rad_cutoff.powf(9.0));
    for ips in 0..ps.len() {
        let mut sum = 0.0;
        for jps in 0..ps.len() {
            if ips == jps {
                continue;
            }
            sum += cubic_square_distance(&ps[ips], &ps[jps], rad_cutoff);
        }
        ps[ips].rho = p_mass * poly6_kern * sum;
        ps[ips].p = int_stiff * (ps[ips].rho - rest_density);
        ps[ips].rho = 1.0 / ps[ips].rho;
    }
}

fn density_pressure_hash(
    ps: &mut Vec<s_particle>,
    rad_cutoff: f32,
    p_mass: f32,
    rest_density: f32,
    stiff: f32,
    sg: &srchgrid::SearchGrid,
) {
    let Poly6Kern = 315.0 / (64.0 * std::f32::consts::PI * rad_cutoff.powf(9.0));
    for ips in 0..ps.len() {
        let mut sum = 0.0;
        let indices = sg.get_one_ring_neighbors(ps[ips].x);
        for jps in indices {
            sum += cubic_square_distance(&ps[ips], &ps[jps], rad_cutoff);
        }
        ps[ips].rho = p_mass * Poly6Kern * sum;
        ps[ips].p = stiff * (ps[ips].rho - rest_density);
        ps[ips].rho = 1.0 / ps[ips].rho;
    }
}

fn force(
    f: &mut nalgebra::Vector3<f32>,
    ps0: &s_particle,
    ps1: &s_particle,
    rad_cutoff: f32,
    spiky_kern: f32,
    vterm: f32,
) {
    let dr = ps0.x - ps1.x;
    let dr_norm = dr.norm();
    if rad_cutoff < dr_norm {
        return;
    } else {
        let c = rad_cutoff - dr_norm;
        let pterm = -0.5 * c * spiky_kern * (ps0.p + ps1.p) / dr_norm;
        let mut fcurr = pterm * dr + vterm * (ps1.v - ps0.v);
        fcurr *= c * ps0.rho * ps1.rho;
        *f += fcurr;
    }
}

fn accumulate_force(ps: &mut Vec<s_particle>, rad_cutoff: f32, visc: f32) {
    let SpikyKern = -45.0 / (std::f32::consts::PI * rad_cutoff.powf(6.0));
    let LapKern = -SpikyKern;
    for ips in 0..ps.len() {
        let mut f = nalgebra::Vector3::zeros();
        for jps in 0..ps.len() {
            if ips == jps {
                continue;
            }
            force(
                &mut f,
                &ps[ips],
                &ps[jps],
                rad_cutoff,
                SpikyKern,
                LapKern * visc,
            );
        }
        ps[ips].f = f;
    }
}

fn update_position(
    ps: &mut Vec<s_particle>,
    p_mass: f32,
    limit: f32,
    ext_stiff: f32,
    ext_damp: f32,
    dt: f32,
    epsilon: f32,
    bbmin: nalgebra::Vector3<f32>,
    bbmax: nalgebra::Vector3<f32>,
    rad_cutoff: f32,
) {
    let gravity = nalgebra::Vector3::new(0.0, -9.8, 0.0);
    for p in ps {
        let mut a = p.f * p_mass;
        let speed = a.norm();
        if speed > limit {
            a *= limit / speed;
        }
        for idim in 0..3 {
            {
                let diff = 2.0 * rad_cutoff - (p.x[idim] - bbmin[idim]);
                if diff > epsilon {
                    let mut norm: nalgebra::Vector3<f32> = nalgebra::Vector3::zeros();
                    norm[idim] = 1.0;
                    let adj = ext_stiff * diff - ext_damp * p.v.dot(&norm);
                    a += adj * norm;
                }
            }
            {
                let diff = 2.0 * rad_cutoff - (bbmax[idim] - p.x[idim]);
                if diff > epsilon {
                    let mut norm: nalgebra::Vector3<f32> = nalgebra::Vector3::zeros();
                    norm[idim] = -1.0;
                    let adj = ext_stiff * diff - ext_damp * p.v.dot(&norm);
                    a += adj * norm;
                }
            }
        }
        a += gravity;
        p.v += a * dt;
        p.x += p.v * dt;
    }
}

fn export_obj(particles: &Vec<s_particle>, filename: String) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    // Write OBJ header (optional, but good practice)
    writeln!(writer, "# Exported from Rust simulation")?;

    // Write vertices
    for particle in particles {
        writeln!(writer, "v {} {} {}", particle.x[0], particle.x[1], particle.x[2])?;
    }

    // Write faces (assuming each particle is a separate vertex, no faces are needed for points)
    //If your particles form a mesh, you'll need to define faces here.
    // For example, if you were simulating a surface mesh, you would add face definitions here.

    Ok(())
}

fn main() {
    let dt = 0.001;
    let rad_cutoff = 0.01;
    let sph_radius = 0.001;
    let epsilon = 1.0e-5;
    let bbmin = nalgebra::Vector3::new(0.0, 0.0, 0.0);
    let bbmax = nalgebra::Vector3::new(0.1, 0.1, 0.05);
    let p_mass: f32 = 0.0001;
    let int_stiff = 1.0;
    let ext_stiff = 50000.0;
    let ext_damp = 256.0;
    let rest_density = 600.0;
    let visc = 0.2;
    let limit = 200.0;

    let mut particles: Vec<s_particle> = Vec::new();
    let mut rng = StdRng::from_entropy(); // Random number generator
    let dist = Uniform::new(-1.0, 1.0); // Uniform distribution [-1, 1]

    // Compute particle spacing
    let d: f32 = (p_mass / rest_density).powf(1.0 / 3.0) * 0.87;

    // Initialization bounds
    let init_min = nalgebra::Vector3::new(0.0, 0.0, 0.0);
    let init_max = nalgebra::Vector3::new(0.05, 0.1, 0.05);

    // Calculate grid divisions
    let ndiv_x = ((init_max.x - init_min.x) / d).ceil() as i32;
    let ndiv_y = ((init_max.y - init_min.y) / d).ceil() as i32;
    let ndiv_z = ((init_max.z - init_min.z) / d).ceil() as i32;
    dbg!(ndiv_x, ndiv_y, ndiv_z);
    // Iterate through grid and create particles
    for idiv_x in 0..ndiv_x {
        for idiv_y in 0..ndiv_y {
            for idiv_z in 0..ndiv_z {
                // Generate random offsets
                let random_offset_x = dist.sample(&mut rng) * 0.3;
                let random_offset_y = dist.sample(&mut rng) * 0.3;
                let random_offset_z = dist.sample(&mut rng) * 0.3;

                // Compute particle position
                let r = nalgebra::Vector3::new(
                    init_min.x + d * (idiv_x as f32 + 0.5 + random_offset_x),
                    init_min.y + d * (idiv_y as f32 + 0.5 + random_offset_y),
                    init_min.z + d * (idiv_z as f32 + 0.5 + random_offset_z),
                );

                // Rejection sampling to ensure particles are within bounds
                if r.x < init_min.x || r.x > init_max.x {
                    continue;
                }
                if r.y < init_min.y || r.y > init_max.y {
                    continue;
                }
                if r.z < init_min.z || r.z > init_max.z {
                    continue;
                }

                // Initialize velocity to zero
                let v = nalgebra::Vector3::zeros();

                // Create and store particle
                particles.push(s_particle{x: r, v: v, rho: 0.0, p: 0.0, f: nalgebra::Vector3::zeros()});
            }
        }
    }

    // Output the number of particles
    println!("Particle size: {}", particles.len());
    for iframe in 0..300 {
        density_pressure(&mut particles, rad_cutoff, p_mass, rest_density, int_stiff);
        accumulate_force(&mut particles, rad_cutoff, visc);
        update_position(
            &mut particles,
            p_mass,
            limit,
            ext_stiff,
            ext_damp,
            dt,
            epsilon,
            bbmin,
            bbmax,
            rad_cutoff,
        );
        export_obj(&particles, "fluid".to_owned() + &iframe.to_string() + ".obj").unwrap();
        dbg!(&particles[0]);
    }
}
