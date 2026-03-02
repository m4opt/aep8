C     Thin wrapper that replicates the L-shell / B-B0 computation from
C     fly_in_nasa_aeap1 (AE8_AP8.f lines 111-171) but returns the Lm
C     and BBo arrays instead of calling get_AE8_AP8_flux.
C
C     This file lives outside the IRBEM source tree, so we hardcode
C     NTIME_MAX and baddata instead of using INCLUDE files.

      SUBROUTINE compute_lbbo(ntime,sysaxes,kint_in,
     &     iyear,idoy,UT,xIN1,xIN2,xIN3,Lm_out,BBo_out)
c
      IMPLICIT NONE
c
c     Constants (match IRBEM values)
      INTEGER*4 NTIME_MAX
      PARAMETER (NTIME_MAX = 100000)
      REAL*8 baddata
      PARAMETER (baddata = -1.d31)
c
c     Inputs
      INTEGER*4  ntime, sysaxes, kint_in
      INTEGER*4  iyear(NTIME_MAX), idoy(NTIME_MAX)
      REAL*8     UT(NTIME_MAX)
      REAL*8     xIN1(NTIME_MAX), xIN2(NTIME_MAX), xIN3(NTIME_MAX)
c
c     Outputs
      REAL*8     Lm_out(NTIME_MAX), BBo_out(NTIME_MAX)
c
c     Internal variables
      INTEGER*4  k_ext, k_l, kint, isat
      INTEGER*4  t_resol, r_resol, Ilflag
      REAL*8     xGEO(3)
      REAL*8     alti, lati, longi, psi, tilt
      REAL*8     ERA, AQUAD, BQUAD
      REAL*8     BLOCAL, BMIN, XJ, Lstar
      REAL*8     pi, rad
c
      COMMON /GENER/ ERA, AQUAD, BQUAD
      COMMON /magmod/ k_ext, k_l, kint
      COMMON /flag_L/ Ilflag
      COMMON /dip_ang/ tilt
      COMMON /rconst/ rad, pi
c
c     Set up magnetic field model parameters
      Ilflag = 0
      k_ext = 0
      t_resol = 3
      r_resol = 0
      k_l = 0
      kint = kint_in
c
      CALL INITIZE
      if (kint .eq. 2) then
         CALL JensenANDCain1960
      endif
      if (kint .eq. 3) then
         CALL GSFC1266
      endif
c
c     Loop over all time/position points
      DO isat = 1, ntime
c        Reinitialize sun direction for coordinate transforms
         CALL INIT_GSM(iyear(isat), idoy(isat), UT(isat), psi)
         tilt = psi / rad
c
         call get_coordinates(sysaxes,
     &        xIN1(isat), xIN2(isat), xIN3(isat),
     &        alti, lati, longi, xGEO)
c
         CALL calcul_Lstar_opt(t_resol, r_resol, xGEO,
     &        Lm_out(isat), Lstar, XJ, BLOCAL, BMIN)
c
         if (Lm_out(isat) .le. 0.D0 .and.
     &       Lm_out(isat) .ne. baddata)
     &       Lm_out(isat) = -Lm_out(isat)
c
c        Use McIlwain Gmagmo for B0 calculation (matches fly_in_nasa_aeap1)
         BBo_out(isat) = BLOCAL / (31165.3 / Lm_out(isat)**3)
      ENDDO
c
      END
