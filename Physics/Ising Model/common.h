#ifndef PARAMETERS_H
#define PARAMETERS_H 1

#include <stdint.h>
#include <stdbool.h>

// Must match the data types in ising.py
typedef int8_t spin_t;
typedef int16_t index_t;

typedef struct
{
	bool plt_thermodynamic_functions, plt_thermodynamic_functions_for_different_h,
	     plt_domains, plt_conv_inf_iter, plt_conv_inf_lattice, anim_state_transitions;
} flags_t;

typedef struct
{
	flags_t flag;
	double s_0, t_anim;
	int32_t n_frames, n_threads, n_h_norm, n_t_domain, n_t, n_rows, n_cols, n_spins, n_pass;
	int64_t n_iter_0;
	int32_t n_n_iter_mult, n_lattice_size_mult;
} parameters_t;

typedef struct
{
	spin_t *domain, *lattice_anim;
	double *h_norm, *t_domain, *t, *s_f, *s_avg, *u_f, *n_iter_mult, *lattice_size_mult, *s_f_i,
	       *s_f_l;
} pointers_t;

extern parameters_t _param;

#define LATTICE2D(lattice) spin_t (*lattice)[_param.n_cols]
#define LATTICE2D_ARRAY(lattice) spin_t (*lattice)[_param.n_rows][_param.n_cols]
#define DOUBLE2D_ARRAY(array) double (*array)[_param.n_t]

#endif
