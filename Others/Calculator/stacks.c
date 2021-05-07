#include "definitions.c"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <complex.h>
#include <string.h>

void push_calc_stack(void *);
void pop_calc_stack(void);
void push_num_stack(void *);

void push_calc_stack(void *numptr)
{
	extern int offset;
	extern struct calc_stack_ calc_stack;
	calc_stack.ptr = realloc(calc_stack.ptr, (++calc_stack.index + 1)*offset);
	memcpy((void *)((char *)calc_stack.ptr + calc_stack.index*offset), numptr, (size_t)offset);
	return;
}

void pop_calc_stack(void)
{
	extern int offset;
	extern struct calc_stack_ calc_stack;
	calc_stack.ptr = realloc(calc_stack.ptr, calc_stack.index*offset);
	--calc_stack.index;
	return;
}

void push_num_stack(void *numptr)
{
	extern int offset;
	extern struct num_stack_ num_stack;
	num_stack.ptr = realloc(num_stack.ptr, (++num_stack.index + 1)*offset);
	memcpy((void *)((char *)num_stack.ptr + num_stack.index*offset), numptr, (size_t)offset);
	return;
}

void *getnum_and_update_index(char *input, int *intptr)
{
	extern enum number_system_ number_system;
	static void *vptr = NULL;
	free(vptr);
	char *endptr;
	if(number_system == real_)
	{
		vptr = malloc(sizeof(double));
		*((double *)vptr) = strtod(input - 1, &endptr);
	} else if(number_system == complex_)
	{
		vptr = malloc(sizeof(complex double));
		*((complex double *)vptr) = strtod(input - 1, &endptr);
		*((complex double *)vptr) += strtod(endptr, &endptr)*1i;
	} else
	{
		exit(EXIT_FAILURE);
	}
	*intptr += abs((int)((uintptr_t)endptr - (uintptr_t)input));
	return vptr;
}

void gen_num_stack(char *input)
{
	extern struct num_stack_ num_stack;
	for(int i = 0; input[i] != '\0'; ++i)
	{
		if(isdigit(input[i]))
		{
			num_stack.push(getnum_and_update_index(input + i, &i));
		}
	}
	num_stack.index = 0;
	return;
}
