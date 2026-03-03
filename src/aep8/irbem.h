// External functions and common blocks from IRBEM Fortran code.
#include <stdint.h>

extern void initize_(void);
extern void jensenandcain1960_(void);
extern void gsfc1266_(void);
extern void init_gsm_(const int *iyr, const int *iday, const double *secs, double *psi);
extern void get_coordinates_(const int *sysaxes, const double *x1, const double *x2, const double *x3, double *alti, double *lati, double *longi, double xGEO[3]);
extern void calcul_lstar_opt_(const int *t_resol, const int *r_resol, const double xGEO[3], double *Lm, double *Lstar, double *leI0, double *B0, double *Bmin);
extern void init_ae8min_(void);
extern void init_ae8max_(void);
extern void init_ap8min_(void);
extern void init_ap8max_(void);
extern void trara_(const int32_t descr[8], const int *map, double *fl, const double *bb0, const double *e, double *f, const int *whichm);

extern struct {
    int32_t k_ext, k_l, kint;
} magmod_;

extern struct {
    int32_t Ilflag;
} flag_l_;

extern struct {
    double tilt;
} dip_ang_;

extern struct {
    double pi, rad;
} rconst_;

extern struct trara_data {
    int32_t ihead[8], map[];
} promin_, promax_, elemin_, elemax_;
