#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _cacum_reg();
extern void _cagk_reg();
extern void _cal2_reg();
extern void _can2_reg();
extern void _cat_reg();
extern void _ch_CavL_reg();
extern void _ch_CavN_reg();
extern void _ch_HCNolm_reg();
extern void _ch_KCaS_reg();
extern void _ch_Kdrfast_reg();
extern void _ch_KvA_reg();
extern void _ch_KvAolm_reg();
extern void _ch_KvCaB_reg();
extern void _ch_Nav_reg();
extern void _ch_Navaxonp_reg();
extern void _ch_leak_reg();
extern void _h_reg();
extern void _ipulse1_reg();
extern void _kad_reg();
extern void _kaprox_reg();
extern void _kca_reg();
extern void _kdrca1_reg();
extern void _kmb_reg();
extern void _naxn_reg();
extern void _xtra_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," cacum.mod");
fprintf(stderr," cagk.mod");
fprintf(stderr," cal2.mod");
fprintf(stderr," can2.mod");
fprintf(stderr," cat.mod");
fprintf(stderr," ch_CavL.mod");
fprintf(stderr," ch_CavN.mod");
fprintf(stderr," ch_HCNolm.mod");
fprintf(stderr," ch_KCaS.mod");
fprintf(stderr," ch_Kdrfast.mod");
fprintf(stderr," ch_KvA.mod");
fprintf(stderr," ch_KvAolm.mod");
fprintf(stderr," ch_KvCaB.mod");
fprintf(stderr," ch_Nav.mod");
fprintf(stderr," ch_Navaxonp.mod");
fprintf(stderr," ch_leak.mod");
fprintf(stderr," h.mod");
fprintf(stderr," ipulse1.mod");
fprintf(stderr," kad.mod");
fprintf(stderr," kaprox.mod");
fprintf(stderr," kca.mod");
fprintf(stderr," kdrca1.mod");
fprintf(stderr," kmb.mod");
fprintf(stderr," naxn.mod");
fprintf(stderr," xtra.mod");
fprintf(stderr, "\n");
    }
_cacum_reg();
_cagk_reg();
_cal2_reg();
_can2_reg();
_cat_reg();
_ch_CavL_reg();
_ch_CavN_reg();
_ch_HCNolm_reg();
_ch_KCaS_reg();
_ch_Kdrfast_reg();
_ch_KvA_reg();
_ch_KvAolm_reg();
_ch_KvCaB_reg();
_ch_Nav_reg();
_ch_Navaxonp_reg();
_ch_leak_reg();
_h_reg();
_ipulse1_reg();
_kad_reg();
_kaprox_reg();
_kca_reg();
_kdrca1_reg();
_kmb_reg();
_naxn_reg();
_xtra_reg();
}
