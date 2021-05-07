#include "definitions.c"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int is_equal_str(char *, char *);

int choose_number_system(char *str)
{
	extern enum number_system_ number_system;
	extern int offset;
	int isreal = is_equal_str(str, "real"), iscomplex = is_equal_str(str, "complex");
	number_system += (real_ - number_system)*isreal + (complex_ - number_system)*iscomplex;
	offset += ((int)sizeof(double) - offset)*isreal + (int)(sizeof(complex double) - offset)*iscomplex;
	return EXIT_SUCCESS*(isreal || iscomplex) + EXIT_FAILURE*!(isreal || iscomplex);
}

char *get_line(void)
{
	static char *str = NULL;
	free(str);
	int size = 1;
	str = (char *)malloc(size);
	printf("=> ");
	while((str[size - 1] = getchar()) != '\n')
	{
		str = (char *)realloc((void *)str, ++size);
	}
	str[size - 1] = '\0';
	return str;
}

void reset_global_vars(void)
{
	extern struct calc_stack_ calc_stack;
	extern enum operator_ *operator;
	extern struct num_stack_ num_stack;
	extern enum operator_ *expression;
	extern int expression_index;
	free(num_stack.ptr);
	num_stack.ptr = NULL;
	free((void *)operator);
	operator = NULL;
	free(calc_stack.ptr);
	calc_stack.ptr = NULL;
	free((void *)expression);
	expression = NULL;

	num_stack.index = -1;
	calc_stack.index = -1;
	expression_index = -1;
}

void gen_result(void);
void gen_operator_sequence(char *input);
void gen_num_stack(char *);

void run_calc()
{
	char *input = NULL;
	while(!is_equal_str((input = get_line()), "exit"))
	{
		if(choose_number_system(input) == EXIT_SUCCESS){continue;}
		gen_num_stack(input);
		gen_operator_sequence(input);
		gen_result();
		reset_global_vars();
	}
}

int main(int argc, char **argv)
{
	if(argc == 2){choose_number_system(argv[1]);}
	run_calc();
	return 0;
}
