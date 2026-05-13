/**
 * Combined forward simulation: dust emission + encounter detection.
 *
 * Loads a REBOUND .bin setup file, integrates forward while emitting dust
 * near perihelion, writes orbital elements to orbital_elements.h5 (periodic),
 * records near-miss encounters with Earth/Mars/Jupiter to
 * <base>_intersections.h5, and saves a SimulationArchive snapshot with a
 * companion beta/birth_time file once dust emission stops.
 *
 * Usage: ./rebound_combined <setup_file.bin> <sim_days>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rebound.h"
#include <hdf5.h>

#define MAX_PARTICLES        100005
#define AU_IN_M              1.495978707e11
#define C_SI                 299792458.0
#define DAY                  86400.0
#define TWO_PI               (2.0 * M_PI)

static double BETA_LIST[]          = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2};
static int    N_BETA               = 5;
static double EJECTION_V_SIGMA_MPS = 200.0;

/* Per-particle data stored in particle.ap for all dust */
typedef struct {
    double beta;
    double birth_time; /* Julian date of emission */
} DustData;

typedef struct {
    double dt;
    double output_interval;
    int    n_particles;
    int    scale_dt;
} SimConfig;

static SimConfig sim_config = {
    .dt              = 10.0,
    .output_interval = 365.0,
    .n_particles     = 10,
    .scale_dt        = 1
};

/* ---- Orbital-elements HDF5 record ---- */
typedef struct {
    double   time;
    uint32_t particle_id;  /* stable hash: 0-9 = Sun/planets/comet, >=10 = dust */
    double a, e, inc, Omega, omega, M;
    double x, y, z;
    double vx, vy, vz;
    double beta;
    double birth_time;
} OrbElement;

/* ---- Encounter/intersection HDF5 record ---- */
typedef struct {
    double time;
    int    dust_id;
    int    planet_id;
    double dust_x,    dust_y,    dust_z;
    double dust_vx,   dust_vy,   dust_vz;
    double planet_x,  planet_y,  planet_z;
    double planet_vx, planet_vy, planet_vz;
    double beta;
    double birth_time;
    double distance;
    double relative_speed;
} EncounterRecord;

/* ---- Global HDF5 handles ---- */
static hid_t   orbit_file_id  = -1, orbit_dset_id  = -1;
static hid_t   orbit_space_id = -1, orbit_dtype_id = -1;
static hsize_t orbit_nrows    = 0;
static double  min_output_time  = 0.0;
static double  last_output_time = -1.0;

static hid_t   enc_file_id  = -1, enc_dset_id  = -1;
static hid_t   enc_space_id = -1, enc_dtype_id = -1;
static hsize_t enc_nrows    = 0;

/* ---- Simulation state ---- */
static double t_jd_start            = 0.0;
static double t_1900                = 0.0;
static int    dust_emission_stopped = 0;
static int    comet_entered_inner   = 0;
static double prev_r_comet          = -1.0;
static int    r_was_increasing      = -1;
static int    comet_past_perihelion = 0;
static int      g_first_dust_idx    = -1;  /* index of first dust particle */
static long     n_removed_total     = 0;
static uint32_t g_next_dust_hash    = 10;  /* monotone counter; planets fixed at 0-9 */

/* ---- Forward declarations ---- */
void heartbeat(struct reb_simulation *r);
void force_radiation(struct reb_simulation *r);
void cross_product(double *a, double *b, double *result);
double vector_norm(double *v);
void normalize(double *v);
void release_dust_at_comet(struct reb_simulation *sim, int comet_idx, double beta,
                            int n_particles, double sigma_eject_ms, int add_ejection);
void remove_escaped_dust(struct reb_simulation *sim);

/* ---- Config ---- */
void load_config(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) return;
    printf("Loading config: %s\n", filename);
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\0' || *p == '\n') continue;
        char key[64] = {0}, val[64] = {0};
        if (sscanf(line, " %63[^=] = %63s", key, val) != 2) continue;
        char *end = key + strlen(key) - 1;
        while (end >= key && (*end == ' ' || *end == '\t')) *end-- = '\0';
        if      (strcmp(key, "dt")              == 0) { sim_config.dt              = atof(val); printf("  dt = %g\n",              sim_config.dt); }
        else if (strcmp(key, "scale_dt")        == 0) { sim_config.scale_dt        = atoi(val); printf("  scale_dt = %d\n",        sim_config.scale_dt); }
        else if (strcmp(key, "output_interval") == 0) { sim_config.output_interval = atof(val); printf("  output_interval = %g\n", sim_config.output_interval); }
        else if (strcmp(key, "n_particles")     == 0) { sim_config.n_particles     = atoi(val); printf("  n_particles = %d\n",     sim_config.n_particles); }
        else fprintf(stderr, "  warning: unknown config key '%s'\n", key);
    }
    fclose(f);
}

/* ---- Radiation pressure (Poynting-Robertson, Burns et al. 1979 eq.5) ---- */
void force_radiation(struct reb_simulation *r)
{
    struct reb_particle *particles = r->particles;
    const struct reb_particle star = particles[0];
    const int N = r->N;
    const double G = r->G;
#pragma omp parallel for
    for (int i = 4; i < N; i++) {
        const struct reb_particle p = particles[i];
        if (p.m != 0. || p.ap == NULL) continue;
        double beta = ((DustData *)p.ap)->beta;
        if (beta == 0.0) continue;
        const double prx  = p.x  - star.x,  pry  = p.y  - star.y,  prz  = p.z  - star.z;
        const double pr   = sqrt(prx*prx + pry*pry + prz*prz);
        const double prvx = p.vx - star.vx, prvy = p.vy - star.vy, prvz = p.vz - star.vz;
        const double c    = 1.006491504759635e+04;
        const double rdot = (prvx*prx + prvy*pry + prvz*prz) / pr;
        const double F_r  = beta * G * star.m / (pr * pr);
        particles[i].ax += F_r * ((1. - rdot/c) * prx/pr - prvx/c);
        particles[i].ay += F_r * ((1. - rdot/c) * pry/pr - prvy/c);
        particles[i].az += F_r * ((1. - rdot/c) * prz/pr - prvz/c);
    }
}

/* ---- HDF5: orbital elements (time-series) ---- */
void open_orbit_hdf5(const char *filename)
{
    orbit_dtype_id = H5Tcreate(H5T_COMPOUND, sizeof(OrbElement));
    H5Tinsert(orbit_dtype_id, "time",        HOFFSET(OrbElement, time),        H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "particle_id", HOFFSET(OrbElement, particle_id), H5T_NATIVE_UINT);
    H5Tinsert(orbit_dtype_id, "a",           HOFFSET(OrbElement, a),           H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "e",           HOFFSET(OrbElement, e),           H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "inc",         HOFFSET(OrbElement, inc),         H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "Omega",       HOFFSET(OrbElement, Omega),       H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "omega",       HOFFSET(OrbElement, omega),       H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "M",           HOFFSET(OrbElement, M),           H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "x",           HOFFSET(OrbElement, x),           H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "y",           HOFFSET(OrbElement, y),           H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "z",           HOFFSET(OrbElement, z),           H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "vx",          HOFFSET(OrbElement, vx),          H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "vy",          HOFFSET(OrbElement, vy),          H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "vz",          HOFFSET(OrbElement, vz),          H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "beta",        HOFFSET(OrbElement, beta),        H5T_NATIVE_DOUBLE);
    H5Tinsert(orbit_dtype_id, "birth_time",  HOFFSET(OrbElement, birth_time),  H5T_NATIVE_DOUBLE);

    hsize_t init_dims[1] = {0}, max_dims[1] = {H5S_UNLIMITED}, chunk_dims[1] = {4096};
    orbit_space_id = H5Screate_simple(1, init_dims, max_dims);
    hid_t prop_id  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(prop_id, 1, chunk_dims);
    orbit_file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    orbit_dset_id = H5Dcreate(orbit_file_id, "/orbital_elements", orbit_dtype_id,
                               orbit_space_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);
    H5Pclose(prop_id);
    orbit_nrows = 0;
}

void close_orbit_hdf5(void)
{
    H5Dclose(orbit_dset_id);
    H5Sclose(orbit_space_id);
    H5Tclose(orbit_dtype_id);
    H5Fclose(orbit_file_id);
}

void output_orbital_elements(struct reb_simulation *r)
{
    int N = r->N;
    int include_dust = (r->t >= min_output_time);
    OrbElement *buf = (OrbElement *)malloc(N * sizeof(OrbElement));
    if (!buf) { fprintf(stderr, "output_orbital_elements: malloc failed\n"); return; }

    int n_written = 0;
    for (int i = 0; i < N; i++) {
        int is_dust = (r->particles[i].ap != NULL);
        if (is_dust && !include_dust) continue;
        struct reb_orbit o = reb_orbit_from_particle(r->G, r->particles[i], r->particles[0]);
        OrbElement *row    = &buf[n_written++];
        row->time          = t_jd_start + r->t;
        row->particle_id   = r->particles[i].hash;
        row->a             = o.a;
        row->e             = o.e;
        row->inc           = o.inc   * 180.0 / M_PI;
        row->Omega         = o.Omega  * 180.0 / M_PI;
        row->omega         = o.omega  * 180.0 / M_PI;
        row->M             = o.M      * 180.0 / M_PI;
        row->x             = r->particles[i].x;
        row->y             = r->particles[i].y;
        row->z             = r->particles[i].z;
        row->vx            = r->particles[i].vx;
        row->vy            = r->particles[i].vy;
        row->vz            = r->particles[i].vz;
        row->beta          = is_dust ? ((DustData *)r->particles[i].ap)->beta       : 0.0;
        row->birth_time    = is_dust ? ((DustData *)r->particles[i].ap)->birth_time : 0.0;
    }
    if (n_written == 0) { free(buf); return; }

    hsize_t new_size[1] = {orbit_nrows + (hsize_t)n_written};
    H5Dset_extent(orbit_dset_id, new_size);
    hid_t fspace = H5Dget_space(orbit_dset_id);
    hsize_t offset[1] = {orbit_nrows}, count[1] = {(hsize_t)n_written};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t mspace = H5Screate_simple(1, count, NULL);
    H5Dwrite(orbit_dset_id, orbit_dtype_id, mspace, fspace, H5P_DEFAULT, buf);
    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Fflush(orbit_file_id, H5F_SCOPE_LOCAL);
    orbit_nrows += (hsize_t)n_written;
    free(buf);
}

void write_final_orbital_elements(struct reb_simulation *r, const char *filename)
{
    int N = r->N;
    hid_t dtype_id = H5Tcreate(H5T_COMPOUND, sizeof(OrbElement));
    H5Tinsert(dtype_id, "time",        HOFFSET(OrbElement, time),        H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "particle_id", HOFFSET(OrbElement, particle_id), H5T_NATIVE_UINT);
    H5Tinsert(dtype_id, "a",           HOFFSET(OrbElement, a),           H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "e",           HOFFSET(OrbElement, e),           H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "inc",         HOFFSET(OrbElement, inc),         H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "Omega",       HOFFSET(OrbElement, Omega),       H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "omega",       HOFFSET(OrbElement, omega),       H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "M",           HOFFSET(OrbElement, M),           H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "x",           HOFFSET(OrbElement, x),           H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "y",           HOFFSET(OrbElement, y),           H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "z",           HOFFSET(OrbElement, z),           H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "vx",          HOFFSET(OrbElement, vx),          H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "vy",          HOFFSET(OrbElement, vy),          H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "vz",          HOFFSET(OrbElement, vz),          H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "beta",        HOFFSET(OrbElement, beta),        H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_id, "birth_time",  HOFFSET(OrbElement, birth_time),  H5T_NATIVE_DOUBLE);

    OrbElement *buf = (OrbElement *)malloc(N * sizeof(OrbElement));
    if (!buf) { fprintf(stderr, "write_final_orbital_elements: malloc failed\n"); H5Tclose(dtype_id); return; }
    for (int i = 0; i < N; i++) {
        int is_dust        = (r->particles[i].ap != NULL);
        struct reb_orbit o = reb_orbit_from_particle(r->G, r->particles[i], r->particles[0]);
        buf[i].time        = t_jd_start + r->t;
        buf[i].particle_id = r->particles[i].hash;
        buf[i].a           = o.a;
        buf[i].e           = o.e;
        buf[i].inc         = o.inc   * 180.0 / M_PI;
        buf[i].Omega       = o.Omega  * 180.0 / M_PI;
        buf[i].omega       = o.omega  * 180.0 / M_PI;
        buf[i].M           = o.M      * 180.0 / M_PI;
        buf[i].x           = r->particles[i].x;
        buf[i].y           = r->particles[i].y;
        buf[i].z           = r->particles[i].z;
        buf[i].vx          = r->particles[i].vx;
        buf[i].vy          = r->particles[i].vy;
        buf[i].vz          = r->particles[i].vz;
        buf[i].beta        = is_dust ? ((DustData *)r->particles[i].ap)->beta       : 0.0;
        buf[i].birth_time  = is_dust ? ((DustData *)r->particles[i].ap)->birth_time : 0.0;
    }
    hsize_t dims[1] = {(hsize_t)N};
    hid_t space_id  = H5Screate_simple(1, dims, NULL);
    hid_t file_id   = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dset_id   = H5Dcreate(file_id, "/orbital_elements", dtype_id,
                                 space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, dtype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset_id); H5Sclose(space_id); H5Tclose(dtype_id); H5Fclose(file_id);
    free(buf);
    printf("Final orbital elements written to %s (%d particles, JD=%.1f)\n",
           filename, N, t_jd_start + r->t);
}

/* ---- HDF5: encounters ---- */
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

    hsize_t init_dims[1] = {0}, max_dims[1] = {H5S_UNLIMITED}, chunk_dims[1] = {4096};
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
    hsize_t offset[1] = {enc_nrows}, count[1] = {(hsize_t)n};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);
    hid_t mspace = H5Screate_simple(1, count, NULL);
    H5Dwrite(enc_dset_id, enc_dtype_id, mspace, fspace, H5P_DEFAULT, buf);
    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Fflush(enc_file_id, H5F_SCOPE_LOCAL);
    enc_nrows += (hsize_t)n;
}

void check_and_record_encounters(struct reb_simulation *r)
{
    static const int TARGET_PLANETS[] = {3, 4, 5}; /* Earth, Mars, Jupiter */
    static const int N_TARGETS = 3;
    int N = r->N;
    double dt = r->dt;

    int capacity = 256, n_enc = 0;
    EncounterRecord *buf = (EncounterRecord *)malloc(capacity * sizeof(EncounterRecord));
    if (!buf) { fprintf(stderr, "check_and_record_encounters: malloc failed\n"); return; }

    for (int i = 0; i < N; i++) {
        if (r->particles[i].ap == NULL) continue;
        DustData *dd             = (DustData *)r->particles[i].ap;
        struct reb_particle *dust = &r->particles[i];

        for (int t = 0; t < N_TARGETS; t++) {
            int pidx = TARGET_PLANETS[t];
            if (pidx >= N) continue;
            struct reb_particle *planet = &r->particles[pidx];

            double dx = dust->x - planet->x, dy = dust->y - planet->y, dz = dust->z - planet->z;
            double dist  = sqrt(dx*dx + dy*dy + dz*dz);
            double dvx   = dust->vx - planet->vx, dvy = dust->vy - planet->vy, dvz = dust->vz - planet->vz;
            double V_rel = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

            if (dist < V_rel * dt) {
                if (n_enc == capacity) {
                    capacity *= 2;
                    EncounterRecord *tmp = (EncounterRecord *)realloc(buf, capacity * sizeof(EncounterRecord));
                    if (!tmp) { fprintf(stderr, "check_and_record_encounters: realloc failed\n"); free(buf); return; }
                    buf = tmp;
                }
                EncounterRecord *rec = &buf[n_enc++];
                rec->time           = t_jd_start + r->t;
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
    if (n_enc > 0) append_encounter_records(buf, n_enc);
    free(buf);
}

/* Save SimulationArchive + companion beta/birth_time file once dust emission stops. */
void save_archive_snapshot(struct reb_simulation *sim, const char *setup_file)
{
    char base[512];
    strncpy(base, setup_file, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';
    size_t len = strlen(base);
    if (len > 4 && strcmp(base + len - 4, ".bin") == 0)
        base[len - 4] = '\0';

    char archive_path[600], betas_path[600];
    snprintf(archive_path, sizeof(archive_path), "%s_snapshot.bin",       base);
    snprintf(betas_path,   sizeof(betas_path),   "%s_snapshot_betas.txt", base);

    reb_simulation_save_to_file(sim, archive_path);
    printf("SimulationArchive saved to '%s' (JD=%.1f, N=%d)\n",
           archive_path, t_jd_start + sim->t, sim->N);

    FILE *f = fopen(betas_path, "w");
    if (!f) { fprintf(stderr, "Warning: could not write beta file '%s'\n", betas_path); return; }
    fprintf(f, "# particle_index beta birth_time\n");
    for (int i = 0; i < sim->N; i++) {
        int    is_dust = (sim->particles[i].ap != NULL);
        double beta    = is_dust ? ((DustData *)sim->particles[i].ap)->beta       : 0.0;
        double bt      = is_dust ? ((DustData *)sim->particles[i].ap)->birth_time : 0.0;
        fprintf(f, "%d %.17g %.17g\n", i, beta, bt);
    }
    fclose(f);
    printf("Beta/birth_time values written to '%s'\n", betas_path);
}

/* ---- Vector utilities ---- */
void cross_product(double *a, double *b, double *result)
{
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
}
double vector_norm(double *v) { return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
void normalize(double *v) { double n = vector_norm(v); if (n > 0) { v[0]/=n; v[1]/=n; v[2]/=n; } }

double randn(void)
{
    static int has_spare = 0;
    static double spare;
    if (has_spare) { has_spare = 0; return spare; }
    has_spare = 1;
    double u, v, s;
    do {
        u = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        v = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

void release_dust_at_comet(struct reb_simulation *sim, int comet_idx, double beta,
                            int n_particles, double sigma_eject_ms, int add_ejection)
{
    if (comet_idx < 0 || comet_idx >= sim->N) {
        fprintf(stderr, "Error: Invalid comet index %d\n", comet_idx);
        return;
    }
    if (sim->N >= MAX_PARTICLES - 1) return;

    struct reb_particle comet = sim->particles[comet_idx];
    double cx = comet.x, cy = comet.y, cz = comet.z;
    double cvx = comet.vx, cvy = comet.vy, cvz = comet.vz;
    double r_comet = sqrt(cx*cx + cy*cy + cz*cz);
    if (!isfinite(r_comet) || r_comet < 1e-3) {
        fprintf(stderr, "Error: Comet state invalid (r=%e)\n", r_comet);
        return;
    }

    double flux_factor      = 2.0 / (r_comet * r_comet);
    int    n_eff            = (int)round(n_particles * flux_factor);
    if (n_eff < 1) n_eff    = 1;
    double pos_scale_au     = fmax(1e-8, 1e-4 * r_comet);
    double beta_factor      = sqrt(beta / 1e-3);
    double sigma_au_per_day = (sigma_eject_ms * flux_factor * beta_factor) * (DAY / AU_IN_M);

    double anti_sun[3] = {cx, cy, cz};
    normalize(anti_sun);
    double tmp[3] = {1, 0, 0};
    if (fabs(anti_sun[0]) >= 0.9) { tmp[0] = 0; tmp[1] = 1; tmp[2] = 0; }
    double u[3], v[3];
    cross_product(tmp, anti_sun, u); normalize(u);
    cross_product(anti_sun, u, v);

    for (int i = 0; i < n_eff; i++) {
        double dvx = 0, dvy = 0, dvz = 0;
        if (add_ejection && sigma_eject_ms > 0) {
            double _r   = sqrt((double)rand() / RAND_MAX);
            double _phi = TWO_PI * ((double)rand() / RAND_MAX);
            double _x = _r * cos(_phi), _y = _r * sin(_phi), _z = sqrt(1 - _r*_r);
            dvx = sigma_au_per_day * (_x*u[0] + _y*v[0] + _z*anti_sun[0]);
            dvy = sigma_au_per_day * (_x*u[1] + _y*v[1] + _z*anti_sun[1]);
            dvz = sigma_au_per_day * (_x*u[2] + _y*v[2] + _z*anti_sun[2]);
        }
        struct reb_particle dust = {0};
        dust.x  = cx  + randn() * pos_scale_au;
        dust.y  = cy  + randn() * pos_scale_au;
        dust.z  = cz  + randn() * pos_scale_au;
        dust.vx = cvx + dvx;
        dust.vy = cvy + dvy;
        dust.vz = cvz + dvz;
        dust.m  = 0.0;
        DustData *dd   = (DustData *)malloc(sizeof(DustData));
        dd->beta       = beta;
        dd->birth_time = t_jd_start + sim->t;
        dust.ap        = dd;
        dust.hash      = g_next_dust_hash++;
        reb_simulation_add(sim, dust);
    }
}

/* Remove dust particles with a < 0 (hyperbolic escape) or a > 100 AU.
 * Iterates backwards so keepSorted=1 removals don't affect unchecked indices. */
void remove_escaped_dust(struct reb_simulation *sim)
{
    if (g_first_dust_idx < 0) return;
    int n_removed = 0;
    for (int i = sim->N - 1; i >= g_first_dust_idx; i--) {
        if (sim->particles[i].ap == NULL) continue;
        struct reb_orbit o = reb_orbit_from_particle(sim->G, sim->particles[i], sim->particles[0]);
        if (o.a < 0.0 || o.a > 100.0) {
            free(sim->particles[i].ap);
            sim->particles[i].ap = NULL;
            reb_simulation_remove_particle(sim, i, 1);
            n_removed++;
        }
    }
    if (n_removed > 0) {
        n_removed_total += n_removed;
        printf("  Removed %d escaped dust particles (a<0 or a>100 AU)  "
               "total removed: %ld  active dust: %d\n",
               n_removed, n_removed_total, sim->N - g_first_dust_idx);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <setup_file.bin> <sim_days>\n", argv[0]);
        return 1;
    }
    char *setup_file = argv[1];
    int   num_days   = atoi(argv[2]);
    load_config("sim.cfg");
    t_1900 = num_days - 365 * 125; // 2025!

    struct reb_simulation *sim = reb_simulation_create_from_file(setup_file, 0);
    if (!sim) { fprintf(stderr, "Error: could not load '%s'\n", setup_file); return 1; }
    printf("Loaded '%s': N=%d, sim->t=%.1f, t_1900=%.1f\n",
           setup_file, sim->N, sim->t, t_1900);

    /* Assign stable hashes to Sun, planets, comet (0 .. N-1) */
    for (int i = 0; i < sim->N; i++)
        sim->particles[i].hash = (uint32_t)i;
    g_next_dust_hash = (uint32_t)sim->N;

    sim->dt                          = sim_config.dt;
    sim->integrator                  = REB_INTEGRATOR_WHFAST;
    sim->force_is_velocity_dependent = 1;
    sim->additional_forces           = force_radiation;
    sim->heartbeat                   = heartbeat;
    sim->N_active                    = sim->N;
    t_jd_start                       = sim->t;
    sim->t                           = 0;
    double tmax                      = num_days;

    min_output_time  = 0.0;
    int    N_steps   = (int)(tmax / 10);
    double dt_step   = tmax / N_steps;
    int    target_index = sim->N - 1;
    g_first_dust_idx    = target_index + 1;

    /* Derive encounter output path from setup_file */
    char base[512];
    strncpy(base, setup_file, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';
    size_t blen = strlen(base);
    if (blen > 4 && strcmp(base + blen - 4, ".bin") == 0)
        base[blen - 4] = '\0';
    char enc_path[600];
    snprintf(enc_path, sizeof(enc_path), "%s_intersections.h5", base);

    reb_simulation_move_to_com(sim);
    open_orbit_hdf5("orbital_elements.h5");
    open_encounter_hdf5(enc_path);
    printf("Writing intersections to '%s'\n", enc_path);

    int archive_saved = 0;

    for (int step = 0; step <= N_steps; step++) {
        reb_simulation_integrate(sim, step * dt_step);

        /* Save archive once, immediately after dust emission stops */
        if (!archive_saved && dust_emission_stopped) {
            printf("Saving archive snapshot at JD=%.1f (%d particles)...\n",
                   t_jd_start + sim->t, sim->N);
            save_archive_snapshot(sim, setup_file);
            archive_saved = 1;
        }

        {
            struct reb_particle target = sim->particles[target_index];
            struct reb_orbit o = reb_orbit_from_particle(sim->G, target, sim->particles[0]);
            double r = sqrt(target.x*target.x + target.y*target.y + target.z*target.z);

            if (r < 3.0) comet_entered_inner = 1;

            if (!dust_emission_stopped) {
                if (comet_entered_inner && r >= 3.0) {
                    dust_emission_stopped = 1;
                    printf("Dust emission stopped: comet exited 3 AU at t=%.1f years\n",
                           sim->t / 365.0);
                }
                if (!dust_emission_stopped && comet_past_perihelion &&
                    prev_r_comet >= 0.0 && r < prev_r_comet && r_was_increasing == 1) {
                    dust_emission_stopped = 1;
                    printf("Dust emission stopped: comet at aphelion at t=%.1f years\n",
                           sim->t / 365.0);
                }
            }

            if (prev_r_comet >= 0.0) {
                int now_increasing = (r > prev_r_comet);
                if (r_was_increasing == 0 && now_increasing) comet_past_perihelion = 1;
                r_was_increasing = now_increasing;
            }
            prev_r_comet = r;

            if (!dust_emission_stopped && sim->N < MAX_PARTICLES - 1) {
                double true_anomaly_deg = fabs(o.f * 180.0 / M_PI);
                if (r < 3.0 && (true_anomaly_deg < 60.0 || true_anomaly_deg > 300.0))
                    for (int bi = 0; bi < N_BETA; bi++)
                        release_dust_at_comet(sim, target_index, BETA_LIST[bi],
                                              sim_config.n_particles, EJECTION_V_SIGMA_MPS, 1);
            }
        }
    }

    write_final_orbital_elements(sim, "final_orbital_elements.h5");
    reb_simulation_free(sim);
    close_orbit_hdf5();
    close_encounter_hdf5();
    printf("Done. %llu encounter records written.\n", (unsigned long long)enc_nrows);
    return 0;
}

void heartbeat(struct reb_simulation *r)
{
    if (reb_simulation_output_check(r, 100.))
        reb_simulation_output_timing(r, INFINITY);

    /* Check encounters every timestep, but only after year 1900 to avoid
     * spurious perihelion-pulse hits during dust injection */
    if (t_1900 < 0 || r->t >= t_1900)
        check_and_record_encounters(r);

    /* Remove escaped dust annually (a < 0 or a > 100 AU) */
    static double last_removal_time = -1.0;
    if (last_removal_time < 0.0 || r->t - last_removal_time >= 365.0) {
        remove_escaped_dust(r);
        last_removal_time = r->t;
    }

    /* Write orbital elements periodically */
    double interval = sim_config.output_interval;
    if (fmod(r->t, interval) < r->dt && fabs(r->t - last_output_time) > 1e-9) {
        output_orbital_elements(r);
        last_output_time = r->t;
    }
}
