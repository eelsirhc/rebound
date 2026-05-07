/**
 * Solar System with test particles
 *
 * This example integrates all planets of the Solar
 * System and dust particles. The initial data comes
 * from the NASA HORIZONS system and was saved to
 * a binary file beforehand. The integrator used is WHFast
 * with a 4 day timestep. Note that close encounters are
 * not resolved. The OpenMP speedup you get depends on the
 * compiler and CPU that you are using.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rebound.h"
#include <hdf5.h>

#define MAX_PARTICLES 50000000
#define NDIM 3
// Constants
#define AU_IN_M 1.495978707e11 // meters
#define C_SI 299792458.0       // m/s
#define DAY 86400.0            // seconds
#define TWO_PI (2.0 * M_PI)
#define J2000_JD 2451545.0         /* Julian Date of 2000-01-01 12:00 TT */


/* Per-particle data stored in particle.ap for all dust particles */
typedef struct
{
    double beta;
    double birth_time;
} DustData;

typedef struct
{
    double dt;
    int n_particles;
    int scale_dt;
} SimConfig;

static SimConfig sim_config = {
    .dt = 10.0,
    .n_particles = 1000,
    .scale_dt = 1
};

void load_config(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f)
        return;

    printf("Loading config: %s\n", filename);
    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\0' || *p == '\n') continue;

        char key[64] = {0}, val[64] = {0};
        if (sscanf(line, " %63[^=] = %63s", key, val) != 2) continue;

        /* trim trailing whitespace from key */
        char *end = key + strlen(key) - 1;
        while (end >= key && (*end == ' ' || *end == '\t')) *end-- = '\0';

        if (strcmp(key, "dt") == 0)
        {
            sim_config.dt = atof(val);
            printf("  dt = %g\n", sim_config.dt);
        }
        else if (strcmp(key, "scale_dt") == 0)
        {
            sim_config.dt = atof(val);
            printf("  scale_dt = %g\n", sim_config.dt);
        }
        else if (strcmp(key, "n_particles") == 0)
        {
            sim_config.n_particles = atoi(val);
            printf("  n_particles = %d\n", sim_config.n_particles);
        }
        else if (strcmp(key, "output_interval") == 0)
        {
            fprintf(stderr, "  warning: output_interval is no longer used (encounter-only output)\n");
        }
        else
        {
            fprintf(stderr, "  warning: unknown config key '%s'\n", key);
        }
    }
    fclose(f);
}

void release_dust_at_comet(struct reb_simulation *sim, int comet_idx, double beta,
                           int n_particles, double sigma_eject_ms, int add_ejection);
void cross_product(double *a, double *b, double *result);
double vector_norm(double *v);
void normalize(double *v);

/* ---------- HDF5 encounter/intersection output ---------- */
typedef struct
{
    double time;
    int dust_id;
    int planet_id;
    double dust_x, dust_y, dust_z;
    double dust_vx, dust_vy, dust_vz;
    double planet_x, planet_y, planet_z;
    double planet_vx, planet_vy, planet_vz;
    double beta;
    double birth_time;
    double distance;
    double relative_speed;
} EncounterRecord;

static hid_t enc_file_id = -1;
static hid_t enc_dset_id = -1;
static hid_t enc_space_id = -1;
static hid_t enc_dtype_id = -1;
static hsize_t enc_nrows = 0;
static double g_t_sim_end = 0.0;  /* sim->t at end of integration (corresponds to J2000) */

void open_encounter_hdf5(const char *filename)
{
    enc_dtype_id = H5Tcreate(H5T_COMPOUND, sizeof(EncounterRecord));
    H5Tinsert(enc_dtype_id, "time",           HOFFSET(EncounterRecord, time),           H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "dust_id",        HOFFSET(EncounterRecord, dust_id),        H5T_NATIVE_INT);
    H5Tinsert(enc_dtype_id, "planet_id",      HOFFSET(EncounterRecord, planet_id),      H5T_NATIVE_INT);
    H5Tinsert(enc_dtype_id, "dust_x",         HOFFSET(EncounterRecord, dust_x),         H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "dust_y",         HOFFSET(EncounterRecord, dust_y),         H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "dust_z",         HOFFSET(EncounterRecord, dust_z),         H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "dust_vx",        HOFFSET(EncounterRecord, dust_vx),        H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "dust_vy",        HOFFSET(EncounterRecord, dust_vy),        H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "dust_vz",        HOFFSET(EncounterRecord, dust_vz),        H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "planet_x",       HOFFSET(EncounterRecord, planet_x),       H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "planet_y",       HOFFSET(EncounterRecord, planet_y),       H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "planet_z",       HOFFSET(EncounterRecord, planet_z),       H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "planet_vx",      HOFFSET(EncounterRecord, planet_vx),      H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "planet_vy",      HOFFSET(EncounterRecord, planet_vy),      H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "planet_vz",      HOFFSET(EncounterRecord, planet_vz),      H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "beta",           HOFFSET(EncounterRecord, beta),           H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "birth_time",     HOFFSET(EncounterRecord, birth_time),     H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "distance",       HOFFSET(EncounterRecord, distance),       H5T_NATIVE_DOUBLE);
    H5Tinsert(enc_dtype_id, "relative_speed", HOFFSET(EncounterRecord, relative_speed), H5T_NATIVE_DOUBLE);

    hsize_t init_dims[1] = {0};
    hsize_t max_dims[1] = {H5S_UNLIMITED};
    hsize_t chunk_dims[1] = {4096};
    enc_space_id = H5Screate_simple(1, init_dims, max_dims);

    hid_t prop_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(prop_id, 1, chunk_dims);

    enc_file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    enc_dset_id = H5Dcreate(enc_file_id, "/intersections", enc_dtype_id,
                            enc_space_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);
    H5Pclose(prop_id);
    enc_nrows = 0;
}

void close_encounter_hdf5(void)
{
    H5Dclose(enc_dset_id);
    H5Sclose(enc_space_id);
    H5Tclose(enc_dtype_id);
    H5Fclose(enc_file_id);
}

static void append_encounter_records(EncounterRecord *buf, int n)
{
    if (n == 0) return;

    hsize_t new_size[1] = {enc_nrows + (hsize_t)n};
    H5Dset_extent(enc_dset_id, new_size);

    hid_t fspace = H5Dget_space(enc_dset_id);
    hsize_t offset[1] = {enc_nrows};
    hsize_t count[1]  = {(hsize_t)n};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    hid_t mspace = H5Screate_simple(1, count, NULL);
    H5Dwrite(enc_dset_id, enc_dtype_id, mspace, fspace, H5P_DEFAULT, buf);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Fflush(enc_file_id, H5F_SCOPE_LOCAL);

    enc_nrows += (hsize_t)n;
}

/* Check every dust particle against Earth (3), Mars (4), Jupiter (5).
 * Record when distance < V_rel * dt (particle within one timestep's reach). */
void check_and_record_encounters(struct reb_simulation *r)
{
    static const int TARGET_PLANETS[] = {3, 4, 5};
    static const int N_TARGETS = 3;

    int N = r->N;
    double dt = r->dt;

    int capacity = 256;
    int n_enc = 0;
    EncounterRecord *buf = (EncounterRecord *)malloc(capacity * sizeof(EncounterRecord));
    if (!buf)
    {
        fprintf(stderr, "check_and_record_encounters: malloc failed\n");
        return;
    }

    for (int i = 0; i < N; i++)
    {
        if (r->particles[i].ap == NULL) continue; /* skip planets and comet */

        DustData *dd = (DustData *)r->particles[i].ap;
        struct reb_particle *dust = &r->particles[i];

        for (int t = 0; t < N_TARGETS; t++)
        {
            int pidx = TARGET_PLANETS[t];
            if (pidx >= N) continue;

            struct reb_particle *planet = &r->particles[pidx];

            double dx = dust->x - planet->x;
            double dy = dust->y - planet->y;
            double dz = dust->z - planet->z;
            double dist = sqrt(dx * dx + dy * dy + dz * dz);

            double dvx = dust->vx - planet->vx;
            double dvy = dust->vy - planet->vy;
            double dvz = dust->vz - planet->vz;
            double V_rel = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);

            if (dist < V_rel * dt)
            {
                if (n_enc == capacity)
                {
                    capacity *= 2;
                    EncounterRecord *tmp = (EncounterRecord *)realloc(buf, capacity * sizeof(EncounterRecord));
                    if (!tmp)
                    {
                        fprintf(stderr, "check_and_record_encounters: realloc failed\n");
                        free(buf);
                        return;
                    }
                    buf = tmp;
                }

                EncounterRecord *rec = &buf[n_enc++];
                rec->time           = J2000_JD + (r->t - g_t_sim_end);
                rec->dust_id        = i;
                rec->planet_id      = pidx;
                rec->dust_x         = dust->x;
                rec->dust_y         = dust->y;
                rec->dust_z         = dust->z;
                rec->dust_vx        = dust->vx;
                rec->dust_vy        = dust->vy;
                rec->dust_vz        = dust->vz;
                rec->planet_x       = planet->x;
                rec->planet_y       = planet->y;
                rec->planet_z       = planet->z;
                rec->planet_vx      = planet->vx;
                rec->planet_vy      = planet->vy;
                rec->planet_vz      = planet->vz;
                rec->beta           = dd->beta;
                rec->birth_time     = dd->birth_time;
                rec->distance       = dist;
                rec->relative_speed = V_rel;
            }
        }
    }

    if (n_enc > 0)
        append_encounter_records(buf, n_enc);

    free(buf);
}

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
        if (p.ap == NULL)
            continue;

        double beta = ((DustData *)p.ap)->beta;
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

/*
 * Release dust particles at the comet's current location.
 *
 * Ejection speed and emission count are scaled by the local solar flux
 * relative to 1 AU (flux_factor = 1/r²), so that:
 *   - sigma_eject is the reference speed at 1 AU; at distance r it becomes
 *     sigma_eject / r²  (r in AU).
 *   - n_particles is the reference emission count at 1 AU; the actual count
 *     emitted at distance r is round(n_particles / r²).
 *
 * Flux scaling: Vaubaillon, Colas & Jorda (2005), A&A 439, 751–760.
 *   "A new method to predict meteor showers. I. Description of the model."
 *   Dust production rate is taken proportional to the heliocentric solar flux,
 *   F ∝ r⁻², giving both the emission rate and the characteristic ejection
 *   energy the same 1/r² dependence.
 */
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

    // Solar flux factor: F(r) / F(1 AU) = 1/r²  (r in AU, already REBOUND units)
    double flux_factor = (2.0) / (r_comet * r_comet); // doubled to 2.0 instead of 1.0 -> normalized to r=1.4...

    // Emission count scaled to local flux; at least 1 if we were called at all
    int n_eff = (int)round(n_particles * flux_factor);
    if (n_eff < 1)
        n_eff = 1;

    double position_perturb_frac = 1e-4;
    double pos_scale_au = fmax(1e-8, position_perturb_frac * r_comet);

    double beta_factor = sqrt(beta/1e-3);
    // Ejection speed scaled by solar flux (1/r²), normalised to sigma_eject_ms at 1 AU
    double sigma_au_per_day = (sigma_eject_ms * flux_factor * beta_factor) * (DAY / AU_IN_M);

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

    for (int i = 0; i < n_eff; i++)
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

        // Store beta and birth_time in particle.ap
        DustData *dd = (DustData *)malloc(sizeof(DustData));
        dd->beta = beta;
        dd->birth_time = sim->t;
        dust.ap = dd;

        reb_simulation_add(sim, dust);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <archive_1900.bin> [years]\n", argv[0]);
        fprintf(stderr, "Example: %s 1P_28124_1900.bin 100\n", argv[0]);
        return 1;
    }

    char *archive_file = argv[1];
    int num_years = 100;

    if (argc > 2)
    {
        num_years = atoi(argv[2]);
        if (num_years <= 0)
        {
            fprintf(stderr, "Error: Number of years must be positive.\n");
            return 1;
        }
    }

    load_config("sim.cfg");

    /* Load the SimulationArchive (last snapshot = year 1900 state) */
    struct reb_simulationarchive *sa = reb_simulationarchive_create_from_file(archive_file);
    if (!sa)
    {
        fprintf(stderr, "Error: Could not open SimulationArchive '%s'\n", archive_file);
        return 1;
    }
    struct reb_simulation *sim = reb_simulation_create_from_simulationarchive(sa, -1);
    reb_simulationarchive_free(sa);
    if (!sim)
    {
        fprintf(stderr, "Error: Could not create simulation from archive\n");
        return 1;
    }
    printf("Loaded archive: N=%d particles, N_active=%d, t=%.1f days\n",
           sim->N, sim->N_active, sim->t);

    /* Derive companion beta file and output file paths from archive name */
    char base[512];
    strncpy(base, archive_file, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';
    size_t blen = strlen(base);
    if (blen > 4 && strcmp(base + blen - 4, ".bin") == 0)
        base[blen - 4] = '\0';

    char betas_path[600], output_path[600];
    snprintf(betas_path,  sizeof(betas_path),  "%s_betas.txt", base);
    snprintf(output_path, sizeof(output_path), "%s_intersections.h5", base);

    /* Restore particle.ap (beta + birth_time) for every dust particle.
     * REBOUND does not serialise void* ap, so all ap fields are NULL after
     * loading — force_radiation and encounter detection both rely on ap != NULL
     * to identify dust, so we must restore it before integrating. */
    FILE *bf = fopen(betas_path, "r");
    if (!bf)
    {
        fprintf(stderr, "Error: Could not open beta file '%s'\n", betas_path);
        fprintf(stderr, "Run problem.c first to generate the 1900 archive and beta file.\n");
        reb_simulation_free(sim);
        return 1;
    }
    char line[256];
    int n_dust_restored = 0;
    while (fgets(line, sizeof(line), bf))
    {
        if (line[0] == '#') continue;
        int idx;
        double beta, birth_time_jd = 0.0;
        int n_parsed = sscanf(line, "%d %lf %lf", &idx, &beta, &birth_time_jd);
        if (n_parsed < 2) continue;
        if (idx < 0 || idx >= sim->N) continue;
        if (beta == 0.0) continue; /* planet or comet — leave ap NULL */
        DustData *dd = (DustData *)malloc(sizeof(DustData));
        dd->beta = beta;
        dd->birth_time = birth_time_jd;
        sim->particles[idx].ap = dd;
        n_dust_restored++;
    }
    fclose(bf);
    printf("Restored beta values for %d dust particles\n", n_dust_restored);

    /* Configure integrator — N_active is preserved from archive */
    sim->dt = sim_config.dt;
    sim->integrator = REB_INTEGRATOR_WHFAST;
    sim->force_is_velocity_dependent = 1;
    sim->additional_forces = force_radiation;
    sim->heartbeat = heartbeat;

    /* Run for num_years starting from the archive time (year 1900) */
    double t_start = sim->t;
    double tmax    = t_start + num_years * 365.0;
    g_t_sim_end = tmax;
    printf("Integrating %.1f years: t=%.1f -> t=%.1f days (JD %.1f -> %.1f)\n",
           (double)num_years, t_start, tmax,
           J2000_JD + (t_start - tmax), J2000_JD);

    int N_steps = (int)((tmax - t_start) / 10.0);
    double dt_step = (tmax - t_start) / N_steps;

    open_encounter_hdf5(output_path);
    printf("Writing intersections to '%s'\n", output_path);

    for (int step = 0; step <= N_steps; step++)
    {
        double t = t_start + step * dt_step;
        reb_simulation_integrate(sim, t);
    }

    reb_simulation_free(sim);
    close_encounter_hdf5();
    printf("Done. %llu encounter records written.\n", (unsigned long long)enc_nrows);
    return 0;
}

void heartbeat(struct reb_simulation *r)
{
    if (reb_simulation_output_check(r, 100.))
    {
        reb_simulation_output_timing(r, INFINITY);
    }
    check_and_record_encounters(r);
}
