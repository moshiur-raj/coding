#include "definitions.c"
#include <stdio.h>
#include <stdlib.h>

enum operator_ *operator = NULL;
enum operator_ *expression = NULL;
int expression_index = -1;
enum number_system_ number_system = real_;
int offset = (int)sizeof(double);

void push_calc_stack(void *);
void pop_calc_stack(void);
struct calc_stack_ calc_stack = {NULL, -1, push_calc_stack, pop_calc_stack};
void push_num_stack(void *);
struct num_stack_ num_stack = {NULL, -1, push_num_stack};

//comparing two strings
int is_equal_str(const char *str1, const char *str2)
{
	int det = 0;
	for(int i = 0; str2[i] != '\0'; ++i)
	{
		det += (str1[i] != str2[i]);
	}
	return !det;
}
