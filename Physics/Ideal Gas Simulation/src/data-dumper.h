#ifndef DATA_DUMPER_H
#define DATA_DUMPER_H 1

#include <variable-types.h>

void dump_data(const vector_t *restrict position, int nmolecules);
void dump_statistical_data(double t, const vector_t *restrict, int nmolecules);

#endif
