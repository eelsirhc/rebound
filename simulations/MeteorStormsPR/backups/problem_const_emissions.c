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
#define OUTPUT_INTERVAL 30.0
// Constants
#define AU_IN_M 1.495978707e11 // meters
#define C_SI 299792458.0       // m/s
#define DAY 86400.0            // seconds
#define TWO_PI (2.0 * M_PI)

static double BETA_LIST[] = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2};
static int N_BETA = 5;
static double EJECTION_V_SIGMA_MPS = 100.0;
// typedef struct
// {
//     int *indices; // particle indices
//     double *beta; // beta values
//     int count;    // current count
//     int capacity; // allocated capacity
// } DustList;
// static DustList dust_list = {NULL, NULL, 0, 0};
void release_dust_at_comet(struct reb_simulation *sim, int comet_idx, double beta,
                           int n_particles, double sigma_eject_ms, int add_ejection);
void cross_product(double *a, double *b, double *result);
double vector_norm(double *v);
void normalize(double *v);

/* ---------- HDF5 orbital-elements output ---------- */
typedef struct
{
    double time;
    int particle_id;
    double a;
    double e;
    double inc;
    double Omega;
    double omega;
    double M;
    double x, y, z;
    double vx, vy, vz;
    double beta;
} OrbElement;

static hid_t orbit_file_id = -1;
static hid_t orbit_dset_id = -1;
static hid_t orbit_space_id = -1;
static hid_t orbit_dtype_id = -1;
static hsize_t orbit_nrows = 0;      /* rows written so far */
static double min_output_time = 0.0; /* minimum time before starting orbital element output */
static double last_output_time = -1.0; /* last time orbital elements were written (prevents duplicates) */

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

        // Get beta from particle.ap (default to 0 if not set)
        double beta = (p.ap != NULL) ? *((double *)p.ap) : 0.0;
        if (beta == 0.0)
            continue;

        const double prx = p.x - star.x;
        const double pry = p.y - star.y;
        const double prz = p.z - star.z;
        const double pr = sqrt(prx * prx + pry * pry + prz * prz); // distance relative to star
        const double prvx = p.vx - star.vx;
        const double prvy = p.vy - star.vy;
        const double prvz = p.vz - star.vz;

        const double c = 1.006491504759635e+04;                          // speed of light.
        const double rdot = (prvx * prx + prvy * pry + prvz * prz) / pr; // radial velocity relative to star
        const double F_r = beta * G * star.m / (pr * pr);

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

/* Create the HDF5 file and an extensible compound dataset for orbital elements. */
void open_orbit_hdf5(const char *filename)
{
    /* Build compound type */
    orbit_dtype_id = H5Tcreate(H5T_COMPOUND, sizeof(OrbElement));
    H5Tinsert(orbit_dtype_id, "time", HOFFSET(OrbElement, time), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "particle_id", HOFFSET(OrbElement, particle_id), H5T_NATIVE_INT);
    H5Tinsert(orbit_dtype_id, "a", HOFFSET(OrbElement, a), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "e", HOFFSET(OrbElement, e), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "inc", HOFFSET(OrbElement, inc), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "Omega", HOFFSET(OrbElement, Omega), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "omega", HOFFSET(OrbElement, omega), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "M", HOFFSET(OrbElement, M), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "x", HOFFSET(OrbElement, x), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "y", HOFFSET(OrbElement, y), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "z", HOFFSET(OrbElement, z), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "vx", HOFFSET(OrbElement, vx), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "vy", HOFFSET(OrbElement, vy), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "vz", HOFFSET(OrbElement, vz), H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "beta", HOFFSET(OrbElement, beta), H5T_NATIVE_DOUBLE);

    /* Create extensible (unlimited) 1-D dataspace with chunking */
    hsize_t init_dims[1] = {0};
    hsize_t max_dims[1] = {H5S_UNLIMITED};
    hsize_t chunk_dims[1] = {4096};
    orbit_space_id = H5Screate_simple(1, init_dims, max_dims);

    hid_t prop_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(prop_id, 1, chunk_dims);

    orbit_file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    orbit_dset_id = H5Dcreate(orbit_file_id, "/orbital_elements", orbit_dtype_id,
                              orbit_space_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);
    H5Pclose(prop_id);
    orbit_nrows = 0;
}

/* Flush and close the orbital-elements HDF5 file. */
void close_orbit_hdf5(void)
{
    H5Dclose(orbit_dset_id);
    H5Sclose(orbit_space_id);
    H5Tclose(orbit_dtype_id);
    H5Fclose(orbit_file_id);
}

/* Append one row per particle to the orbital-elements HDF5 dataset. */
void output_orbital_elements(struct reb_simulation *r)
{
    int N = r->N;
    OrbElement *buf = (OrbElement *)malloc(N * sizeof(OrbElement));
    if (!buf)
    {
        fprintf(stderr, "output_orbital_elements: malloc failed\n");
        return;
    }

    for (int i = 0; i < N; i++)
    {
        struct reb_orbit o = reb_orbit_from_particle(r->G, r->particles[i], r->particles[0]);
        buf[i].time = r->t;
        buf[i].particle_id = i;
        buf[i].a = o.a;
        buf[i].e = o.e;
        buf[i].inc = o.inc * 180.0 / M_PI;
        buf[i].Omega = o.Omega * 180.0 / M_PI;
        buf[i].omega = o.omega * 180.0 / M_PI;
        buf[i].M = o.M * 180.0 / M_PI;
        buf[i].x = r->particles[i].x;
        buf[i].y = r->particles[i].y;
        buf[i].z = r->particles[i].z;
        buf[i].vx = r->particles[i].vx;
        buf[i].vy = r->particles[i].vy;
        buf[i].vz = r->particles[i].vz;
        // Extract beta from particle.ap (0.0 for planets/comet)
        buf[i].beta = (r->particles[i].ap != NULL) ? *((double *)r->particles[i].ap) : 0.0;
    }

    /* Extend the dataset by N rows */
    hsize_t new_size[1] = {orbit_nrows + (hsize_t)N};
    H5Dset_extent(orbit_dset_id, new_size);

    /* Re-open the (now larger) file dataspace and select the new rows */
    hid_t fspace = H5Dget_space(orbit_dset_id);
    hsize_t offset[1] = {orbit_nrows};
    hsize_t count[1] = {(hsize_t)N};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    hid_t mspace = H5Screate_simple(1, count, NULL);
    H5Dwrite(orbit_dset_id, orbit_dtype_id, mspace, fspace, H5P_DEFAULT, buf);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Fflush(orbit_file_id, H5F_SCOPE_LOCAL);

    orbit_nrows += (hsize_t)N;
    free(buf);
}

// Vector operations
void cross_product(double *a, double *b, double *result)
{
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

double vector_norm(double *v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void normalize(double *v)
{
    double norm = vector_norm(v);
    if (norm > 0)
    {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

// Box-Muller transform for normal distribution
double randn()
{
    static int has_spare = 0;
    static double spare;

    if (has_spare)
    {
        has_spare = 0;
        return spare;
    }

    has_spare = 1;
    double u, v, s;
    do
    {
        u = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        v = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

// Release dust particles at comet location
void release_dust_at_comet(struct reb_simulation *sim, int comet_idx, double beta,
                           int n_particles, double sigma_eject_ms, int add_ejection)
{
    if (comet_idx < 0 || comet_idx >= sim->N)
    {
        fprintf(stderr, "Error: Invalid comet index %d\n", comet_idx);
        return;
    }

    struct reb_particle comet = sim->particles[comet_idx];

    // Copy comet state to local variables
    double cx = comet.x, cy = comet.y, cz = comet.z;
    double cvx = comet.vx, cvy = comet.vy, cvz = comet.vz;

    // Sanity check
    double r_comet = sqrt(cx * cx + cy * cy + cz * cz);
    if (!isfinite(r_comet) || r_comet < 1e-3)
    {
        fprintf(stderr, "Error: Comet state invalid (r=%e)\n", r_comet);
        return;
    }

    double position_perturb_frac = 1e-4;
    double pos_scale_au = fmax(1e-8, position_perturb_frac * r_comet);
    double sigma_au_per_day = (sigma_eject_ms) * (DAY / AU_IN_M);

    // Comet-sun vector (anti-solar direction)
    double anti_sun[3] = {cx, cy, cz};
    normalize(anti_sun);

    // Build orthonormal basis for hemisphere
    double tmp[3] = {1, 0, 0};
    if (fabs(anti_sun[0]) >= 0.9)
    {
        tmp[0] = 0;
        tmp[1] = 1;
        tmp[2] = 0;
    }

    double u[3], v[3];
    cross_product(tmp, anti_sun, u);
    normalize(u);
    cross_product(anti_sun, u, v);

    // int before = sim->N;

    for (int i = 0; i < n_particles; i++)
    {
        double dvx = 0, dvy = 0, dvz = 0;

        if (add_ejection && sigma_eject_ms > 0)
        {
            // Cosine-weighted hemisphere sampling
            double _r = sqrt((double)rand() / RAND_MAX);
            double _phi = TWO_PI * ((double)rand() / RAND_MAX);
            double _x = _r * cos(_phi);
            double _y = _r * sin(_phi);
            double _z = sqrt(1 - _r * _r);

            // Direction in inertial frame
            double dx_dir = _x * u[0] + _y * v[0] + _z * anti_sun[0];
            double dy_dir = _x * u[1] + _y * v[1] + _z * anti_sun[1];
            double dz_dir = _x * u[2] + _y * v[2] + _z * anti_sun[2];

            double v_ej = sigma_au_per_day;
            dvx = v_ej * dx_dir;
            dvy = v_ej * dy_dir;
            dvz = v_ej * dz_dir;
        }

        double dx = randn() * pos_scale_au;
        double dy = randn() * pos_scale_au;
        double dz = randn() * pos_scale_au;

        struct reb_particle dust = {0};
        dust.x = cx + dx;
        dust.y = cy + dy;
        dust.z = cz + dz;
        dust.vx = cvx + dvx;
        dust.vy = cvy + dvy;
        dust.vz = cvz + dvz;
        dust.m = 0.0;

        // Store beta in particle.ap
        double *beta_ptr = (double *)malloc(sizeof(double));
        *beta_ptr = beta;
        dust.ap = beta_ptr;

        reb_simulation_add(sim, dust);

        // int new_idx = sim->N - 1;
        // add_dust_particle(new_idx, beta);
    }

    // int added = sim->N - before;
    // printf("Released %d dust particles (beta=%.2e), total particles: %d\n", added, beta, sim->N);
}

int main(int argc, char *argv[])
{
    // Parse command-line arguments for setup file, number of years, and output start year
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <setup_file> [years] [output_start_year]\n", argv[0]);
        fprintf(stderr, "Example: %s Setup.bin 100 75\n", argv[0]);
        return 1;
    }

    char *setup_file = argv[1];
    int num_years = 100;             // default value
    double output_start_year = -1.0; // default to 75% of simulation time

    if (argc > 2)
    {
        num_years = atoi(argv[2]);
        if (num_years <= 0)
        {
            fprintf(stderr, "Error: Number of years must be positive.\n");
            return 1;
        }
    }

    if (argc > 3)
    {
        output_start_year = atof(argv[3]);
        if (output_start_year < 0)
        {
            fprintf(stderr, "Error: Output start year must be non-negative.\n");
            return 1;
        }
    }

    struct reb_simulation *sim = reb_simulation_create_from_file(setup_file, 0);
    printf("%f\n", sim->t);
    // Starting the REBOUND visualization server. This
    // allows you to visualize the simulation by pointing
    // your web browser to http://localhost:1234
    // reb_simulation_start_server(sim, 1234);

    // Setup constants
    sim->dt = 10; // 10.;
    /// 365.25 * 2. * M_PI; // 4days
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
    double tmax = num_years * 365;

    // Set the minimum output time (default to 75% of simulation time)
    if (output_start_year < 0)
    {
        min_output_time = 0.75 * tmax;
    }
    else
    {
        min_output_time = output_start_year * 365;
    }
    int N_steps = tmax / 10;
    double dt = tmax / N_steps;

    int target_index = sim->N - 1;
    // double positions[MAX_PARTICLES][NDIM];

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
    open_orbit_hdf5("orbital_elements.h5");

    for (int step = 0; step <= N_steps; step++)
    {
        double t = step * dt;
        reb_simulation_integrate(sim, t);

        if ((step % 1 == 0) && (sim->N < MAX_PARTICLES - 1))
        {
            struct reb_particle target = sim->particles[target_index];
            // Compute orbital elements to check if near perihelion
            struct reb_orbit o = reb_orbit_from_particle(sim->G, target, sim->particles[0]);

            // Release dust when:
            // 1. Within 3 AU of the sun (active region for most comets)
            // 2. Near perihelion: true anomaly within ±30 degrees of perihelion (f ~ 0)
            double r = sqrt(target.x * target.x + target.y * target.y + target.z * target.z);
            double true_anomaly_deg = fabs(o.f * 180.0 / M_PI);

            // Near perihelion if true anomaly is within ±30° of 0° or 360°
            int near_perihelion = (true_anomaly_deg < 60.0) || (true_anomaly_deg > 300.0);

            if (r < 3.0 && near_perihelion)
                for (int betaindex = 0; betaindex < N_BETA; betaindex++)
                {
                    double beta = BETA_LIST[betaindex];
                    // printf("Releasing beta=%.2e  N=%d\n", beta, n_per_beta);
                    release_dust_at_comet(sim, target_index, beta, 10,
                                          EJECTION_V_SIGMA_MPS, 1);
                }
        }
        //        reb_ensure_bound_to_primary(sim, &new_p);
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

    // // cleanup
    // H5Dclose(dset_id);
    // H5Sclose(space_id);
    // H5Pclose(prop_id);
    // H5Fclose(file_id);
    reb_simulation_free(sim);
    close_orbit_hdf5();
}

void heartbeat(struct reb_simulation *r)
{
    if (reb_simulation_output_check(r, 100.))
    {
        reb_simulation_output_timing(r, INFINITY);
    }
    if (fmod(r->t, OUTPUT_INTERVAL) < r->dt && r->t >= min_output_time)
    {
        // Guard against writing the same timestamp multiple times
        // Use a small epsilon for floating-point comparison
        if (fabs(r->t - last_output_time) > 1e-9)
        {
            output_orbital_elements(r);
            last_output_time = r->t;
        }
    }
}
