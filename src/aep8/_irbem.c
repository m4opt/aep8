#include "irbem.h"
#include "warnings.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>


#ifndef HAVE_EXP10
// exp10 is a GNU extension.
static double exp10(double x) {
    return exp(x * M_LN10);
}
#endif


#ifdef _WIN32
// Definitinon of gmtime_r, a standard C library function missing on Windows.
struct tm *gmtime_r(const time_t *clock, struct tm *result) {
    errno_t err = gmtime_s(result, clock);
    if (err) {
        _set_errno(err);
        return NULL;
    } else {
        return result;
    }
}

// timegm is a GPL and BSD extension.
#define timegm _mkgmtime
#endif


static void must(const char *message, int result) {
    if (result) {
        perror(message);
        abort();
    }
}


static struct geomag_ufunc {
    void (*const init_func)(void);
    const int kint;
    const char name[8];
    const void *data[1];
} geomag_ufuncs[] = {
    {jensenandcain1960_, 2, "geomag3"},
    {gsfc1266_, 3, "geomag4"},
};


// Lock to protect access to Fortran COMMON blocks.
static pthread_once_t geomag_lock_inited = PTHREAD_ONCE_INIT;
static pthread_mutex_t geomag_lock;


static void geomag_lock_init() {
    must("pthread_mutex_init", pthread_mutex_init(&geomag_lock, NULL));
}


static void geomag(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *data
) {
    struct geomag_ufunc *const geomag_ufunc_data = data;

    must("pthread_once", pthread_once(&geomag_lock_inited, geomag_lock_init));
    must("pthread_lock", pthread_mutex_lock(&geomag_lock));

    flag_l_.Ilflag = 0;
    magmod_.k_ext = 0;
    magmod_.k_l = 0;
    magmod_.kint = geomag_ufunc_data->kint;

    initize_();
    geomag_ufunc_data->init_func();

    const npy_intp n = dimensions[0];
    for (npy_intp i = 0; i < n; i ++)
    {
        /* Alignment of the ufunc arguments is enforced by the ufunc API. See
         * https://numpy.org/doc/stable/user/basics.ufuncs.html#use-of-internal-buffers. */
        WARNINGS_PUSH
        WARNINGS_IGNORE_CAST_ALIGN
        time_t epoch = *(time_t *) &args[0][i * steps[0]];
        double    x1 = *(double *) &args[1][i * steps[1]];
        double    x2 = *(double *) &args[2][i * steps[2]];
        double    x3 = *(double *) &args[3][i * steps[3]];
        WARNINGS_POP

        struct tm tmstruct;
        time_t epoch_start_of_day;
        const int sysaxes = 1, t_resol = 3, r_resol = 0;
        int iyr, iday;
        double secs, psi, alti, lati, longi, xGEO[3], Lm, Lstar, BBo, Blocal, Bmin, Xj;

        gmtime_r(&epoch, &tmstruct);
        tmstruct.tm_sec = tmstruct.tm_min = tmstruct.tm_hour = 0;
        epoch_start_of_day = timegm(&tmstruct);

        secs = difftime(epoch, epoch_start_of_day);
        iyr = 1900 + tmstruct.tm_year;
        iday = tmstruct.tm_yday;

        init_gsm_(&iyr, &iday, &secs, &psi);
        dip_ang_.tilt = psi / rconst_.rad;
        get_coordinates_(&sysaxes, &x1, &x2, &x3, &alti, &lati, &longi, xGEO);
        calcul_lstar_opt_(&t_resol, &r_resol, xGEO, &Lm, &Lstar, &Xj, &Blocal, &Bmin);
        if (Lm < 0 && Lm != -1e31) Lm = -Lm;
        BBo = Blocal / (31165.3 / (Lm * Lm * Lm));
        WARNINGS_PUSH
        WARNINGS_IGNORE_CAST_ALIGN
        *(double *) &args[4][i * steps[4]] = BBo;
        *(double *) &args[5][i * steps[5]] = Lm;
        WARNINGS_POP
    }

    must("pthread_unlock", pthread_mutex_unlock(&geomag_lock));
}


static PyUFuncGenericFunction geomag_loops[] = {geomag};
static const char geomag_types[] = {
    NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};


static struct flux_ufunc {
    pthread_once_t inited;
    void (*const init_func)(void);
    struct trara_data *const table;
    const int whichm;
    const double energy_min;
    const double energy_max;
    const char name[6];
    const void *data[1];
} flux_ufuncs[] = {
    {PTHREAD_ONCE_INIT, init_ae8min_, &elemin_, 1, 0.05, 7.0, "flux1"},
    {PTHREAD_ONCE_INIT, init_ae8max_, &elemax_, 2, 0.05, 7.0, "flux2"},
    {PTHREAD_ONCE_INIT, init_ap8min_, &promin_, 3, 0.1, 300.0, "flux3"},
    {PTHREAD_ONCE_INIT, init_ap8max_, &promax_, 4, 0.1, 300.0, "flux4"},
};


// Lock to protect access to Fortran COMMON blocks.
static pthread_once_t flux_lock_inited = PTHREAD_ONCE_INIT;
static pthread_mutex_t flux_lock;


static void flux_lock_init() {
    must("pthread_mutex_init", pthread_mutex_init(&flux_lock, NULL));
}


static void flux(
    char **args,
    const npy_intp *dimensions,
    const npy_intp *steps,
    void *data
) {
    struct flux_ufunc *const flux_ufunc_data = data;

    must("pthread_once", pthread_once(&flux_ufunc_data->inited, flux_ufunc_data->init_func));
    must("pthread_once", pthread_once(&flux_lock_inited, flux_lock_init));
    must("pthread_lock", pthread_mutex_lock(&flux_lock));

    npy_intp n = dimensions[0];
    for (npy_intp i = 0; i < n; i ++)
    {
        /* Alignment of the ufunc arguments is enforced by the ufunc API. See
         * https://numpy.org/doc/stable/user/basics.ufuncs.html#use-of-internal-buffers. */
        WARNINGS_PUSH
        WARNINGS_IGNORE_CAST_ALIGN
        double E = *(double *) &args[0][i * steps[0]];
        double L = *(double *) &args[1][i * steps[1]];
        double B = *(double *) &args[2][i * steps[2]];
        WARNINGS_POP

        double F;
        if (E >= flux_ufunc_data->energy_min && E <= flux_ufunc_data->energy_max) {
            trara_(
                flux_ufunc_data->table->ihead,
                flux_ufunc_data->table->map,
                &L, &B, &E, &F,
                &flux_ufunc_data->whichm);
            F = exp10(F);
        } else {
            F = NPY_NAN;
        }

        WARNINGS_PUSH
        WARNINGS_IGNORE_CAST_ALIGN
        *(double *) &args[3][i * steps[3]] = F;
        WARNINGS_POP
    }

    must("pthread_unlock", pthread_mutex_unlock(&flux_lock));
}


static PyUFuncGenericFunction flux_loops[] = {flux};
static const char flux_types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};


static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_irbem"
};


PyMODINIT_FUNC
PyInit__irbem(void)
{
    #define ADDOBJECT(key, value) do { \
        PyObject *object = value; \
        int result = PyModule_AddObjectRef(module, key, object); \
        Py_XDECREF(object); \
        if (result) { \
            Py_DECREF(module); \
            return NULL; \
        } \
    } while (0)

    import_array();
    import_ufunc();

    PyObject *module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    for (int i = 0; i < sizeof(geomag_ufuncs) / sizeof(geomag_ufuncs[0]); i ++) {
        geomag_ufuncs[i].data[0] = &geomag_ufuncs[i];
        ADDOBJECT(
            geomag_ufuncs[i].name,
            PyUFunc_FromFuncAndData(
                geomag_loops, (void *const *) geomag_ufuncs[i].data, geomag_types,
                1, 4, 2, PyUFunc_None, geomag_ufuncs[i].name, NULL, 0));
    }

    for (int i = 0; i < sizeof(flux_ufuncs) / sizeof(flux_ufuncs[0]); i ++) {
        flux_ufuncs[i].data[0] = &flux_ufuncs[i];
        ADDOBJECT(
            flux_ufuncs[i].name,
            PyUFunc_FromFuncAndData(
                flux_loops, (void *const *) flux_ufuncs[i].data, flux_types,
                1, 3, 1, PyUFunc_None, flux_ufuncs[i].name, NULL, 0));
    }

    return module;
}
