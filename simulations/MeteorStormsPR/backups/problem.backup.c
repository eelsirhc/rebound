/**
 * Solar System with test particles
 *
 * This example integrates all planets of the Solar
 * System and 10000 test particles. The initial data comes
 * from the NASA HORIZONS system and was saved to
 * a binary file beforehand. The integrator used is WHFast
 * with a 4 day timestep. Note that close encounters are
 * not resolved. The OpenMP speedup you get depends on the
 * compiler and CPU that you are using.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rebound.h"
#include <hdf5.h>

#define MAX_PARTICLES 20005
#define NDIM 3
#define OUTPUT_INTERVAL 1.0

FILE *orbit_file;

void heartbeat(struct reb_simulation *r);
void force_radiation(struct reb_simulation *r);
double betaparticles = 0.01; // Beta parameter, defined as the ratio of radiation pressure over gravity.

// Generate random number in [-eps, eps] using REBOUND's RNG
double reb_rand_eps(struct reb_simulation *sim, double eps)
{
    return reb_random_normal(sim, eps);
}
void force_radiation(struct reb_simulation *r)
{
    struct reb_particle *particles = r->particles;
    const struct reb_particle star = particles[0]; // cache
    const int N = r->N;
    const double G = r->G;
#pragma omp parallel for
    for (int i = 4; i < N; i++)
    {
        const struct reb_particle p = particles[i]; // cache
        if (p.m != 0.)
            continue; // Only dust particles feel radiation forces
        const double prx = p.x - star.x;
        const double pry = p.y - star.y;
        const double prz = p.z - star.z;
        const double pr = sqrt(prx * prx + pry * pry + prz * prz); // distance relative to star
        const double prvx = p.vx - star.vx;
        const double prvy = p.vy - star.vy;
        const double prvz = p.vz - star.vz;

        const double c = 1.006491504759635e+04;                          // speed of light.
        const double rdot = (prvx * prx + prvy * pry + prvz * prz) / pr; // radial velocity relative to star
        const double F_r = betaparticles * G * star.m / (pr * pr);

        // Equation (5) of Burns, Lamy, Soter (1979)
        particles[i].ax += F_r * ((1. - rdot / c) * prx / pr - prvx / c);
        particles[i].ay += F_r * ((1. - rdot / c) * pry / pr - prvy / c);
        particles[i].az += F_r * ((1. - rdot / c) * prz / pr - prvz / c);
    }
}

void reb_ensure_bound_to_primary(struct reb_simulation *sim, struct reb_particle *p)
{
    struct reb_particle *primary = &sim->particles[0]; // Central body, e.g., the Sun

    // Relative position
    double dx = p->x - primary->x;
    double dy = p->y - primary->y;
    double dz = p->z - primary->z;
    double r2 = dx * dx + dy * dy + dz * dz;
    double r = sqrt(r2);

    // Relative velocity
    double dvx = p->vx - primary->vx;
    double dvy = p->vy - primary->vy;
    double dvz = p->vz - primary->vz;
    double v2 = dvx * dvx + dvy * dvy + dvz * dvz;

    // Energy per unit mass: E = ½v² - GM/r   (G=1)
    double mu = primary->m;
    double kinetic = 0.5 * v2;
    double potential = -mu / r;
    double energy = kinetic + potential;

    if (energy >= 0.0)
    {
        // Scale velocity to be 90% of escape speed
        double esc2 = 2.0 * mu / r;
        double scale = sqrt(0.9 * esc2 / v2);
        p->vx = primary->vx + dvx * scale;
        p->vy = primary->vy + dvy * scale;
        p->vz = primary->vz + dvz * scale;
    }
}

// Write orbital elements of meteors to CSV
void output_orbital_elements(struct reb_simulation *r)
{
    for (int i = 0; i < r->N; i++)
    { // Skip Sun (0) and comet (1)
        struct reb_orbit o = reb_orbit_from_particle(r->G, r->particles[i], r->particles[0]);
        fprintf(orbit_file, "%.5f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", r->t, i, o.a, o.e, o.inc * 180.0 / M_PI, o.Omega * 180.0 / M_PI, o.omega * 180.0 / M_PI,
                r->particles[i].x, r->particles[i].y, r->particles[i].z, r->particles[i].vx, r->particles[i].vy, r->particles[i].vz);
    }
    fflush(orbit_file);
}

int main(int argc, char *argv[])
{
    struct reb_simulation *sim = reb_simulation_create_from_file("Setup.bin", 0);
    printf("%f\n", sim->t);
    // Starting the REBOUND visualization server. This
    // allows you to visualize the simulation by pointing
    // your web browser to http://localhost:1234
    //    reb_simulation_start_server(sim, 1234);

    // Setup constants
    sim->dt = 4. / 365.25 * 2. * M_PI; // 4days
    sim->integrator = REB_INTEGRATOR_WHFAST;
    //    sim->ri_whfast.coordinates   = REB_WHFAST_COORDINATES_DEMOCRATICHELIOCENTRIC;
    //    sim->integrator                   = REB_INTEGRATOR_IAS15;
    //    sim->ri_ias15.epsilon             = 1e-4;             // accuracy parameter
    //    sim->N_active                     = 1;                // the star is the only massive particle
    sim->force_is_velocity_dependent = 1;
    sim->additional_forces = force_radiation;
    sim->heartbeat = heartbeat;
    sim->N_active = sim->N;
    sim->t = 0;
    double tmax = 100;
    int N_steps = tmax * 10;
    double dt = tmax / N_steps;

    int target_index = sim->N - 1;
    double positions[MAX_PARTICLES][NDIM];

    // HDF5 setup
    // hid_t file_id = H5Fcreate("positions_timeseries.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    // hsize_t dims[3] = {N_steps + 1, MAX_PARTICLES, NDIM}; // Total dataset shape
    // hsize_t chunk_dims[3] = {1, MAX_PARTICLES, NDIM};     // Chunk = one timestep
    // hid_t space_id = H5Screate_simple(3, dims, NULL);

    // hid_t prop_id = H5Pcreate(H5P_DATASET_CREATE);
    // H5Pset_chunk(prop_id, 3, chunk_dims); // Enable chunking

    // hid_t dset_id = H5Dcreate(file_id, "/positions", H5T_NATIVE_DOUBLE,
    //                           space_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);

    reb_simulation_move_to_com(sim);
    // Open output file
    orbit_file = fopen("orbital_elements.csv", "w");
    fprintf(orbit_file, "time,particle_id,a,e,inc_deg,asc_deg,per_deg,x,y,z,vx,vy,vz\n");

    for (int step = 0; step <= N_steps; step++)
    {
        double t = step * dt;
        reb_simulation_integrate(sim, t);

        if ((step % 1 == 0) && (sim->N < MAX_PARTICLES - 1))
        {
            struct reb_particle target = sim->particles[target_index];
            if ((target.x * target.x + target.y * target.y + target.z * target.z) < 30)
            {
                double eps = 1e-3;
                struct reb_particle new_p = target;
                new_p.x *= (1 + reb_rand_eps(sim, eps));
                new_p.y *= (1 + reb_rand_eps(sim, eps));
                new_p.z *= (1 + reb_rand_eps(sim, eps));
                new_p.vx *= (1 + reb_rand_eps(sim, eps));
                new_p.vy *= (1 + reb_rand_eps(sim, eps));
                new_p.vz *= (1 + reb_rand_eps(sim, eps));
                reb_ensure_bound_to_primary(sim, &new_p);
                reb_simulation_add(sim, new_p);
            }
            {
                double eps = 1e-3;
                struct reb_particle new_p = target;
                new_p.x *= (1 + reb_rand_eps(sim, eps));
                new_p.y *= (1 + reb_rand_eps(sim, eps));
                new_p.z *= (1 + reb_rand_eps(sim, eps));
                new_p.vx *= (1 + reb_rand_eps(sim, eps));
                new_p.vy *= (1 + reb_rand_eps(sim, eps));
                new_p.vz *= (1 + reb_rand_eps(sim, eps));
                reb_ensure_bound_to_primary(sim, &new_p);
                reb_simulation_add(sim, new_p);
            }
        }

        // reb_simulation_move_to_com(sim);
        // // Prepare the slice of data
        // for (int i = 0; i < sim->N; i++)
        // {
        //     positions[i][0] = sim->particles[i].x;
        //     positions[i][1] = sim->particles[i].y;
        //     positions[i][2] = sim->particles[i].z;
        // }
        // positions[MAX_PARTICLES - 1][0] = t;
        // positions[MAX_PARTICLES - 1][1] = 0;
        // positions[MAX_PARTICLES - 1][2] = 0;

        // // Write this step as a hyperslab
        // hsize_t offset[3] = {step, 0, 0};            // Start index
        // hsize_t count[3] = {1, MAX_PARTICLES, NDIM}; // Shape of this slice
        // hid_t memspace = H5Screate_simple(3, count, NULL);

        // H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        // H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, space_id, H5P_DEFAULT, positions);

        // H5Sclose(memspace);
    }

    // // cleanup
    // H5Dclose(dset_id);
    // H5Sclose(space_id);
    // H5Pclose(prop_id);
    // H5Fclose(file_id);
    reb_simulation_free(sim);
    fclose(orbit_file);
}

void heartbeat(struct reb_simulation *r)
{
    if (reb_simulation_output_check(r, 100.))
    {
        reb_simulation_output_timing(r, INFINITY);
    }
    if (fmod(r->t, OUTPUT_INTERVAL) < r->dt)
    {
        output_orbital_elements(r);
    }
}
