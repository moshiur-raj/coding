#include "definitions.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <ctype.h>

//push an operator in expression
void update_expression(enum operator_ operator)
{
	extern enum operator_ *expression;
	extern int expression_index;
	expression = (enum operator_ *)realloc((void *)expression, (++expression_index + 1)*sizeof(enum operator_));
	expression[expression_index] = operator;
}

int is_equal_str(char *, char *);

//get operator and update index to next operator
int get_operator_and_update_index(char *cptr)
{
	extern enum number_system_ number_system;
	int i = 0;
	if(((*cptr == '+') || (*cptr == '-')) && isdigit(cptr[1]))
	{
		update_expression(NUMBER);
		//update index until crosses one double var
		do
		{
			++i;
			++cptr;
		} while(isdigit(*cptr) || *cptr == '.');
		if(number_system == complex_)
		{
			//do it again for imaginary part
			do
			{
				++i;
				++cptr;
			} while(isdigit(*cptr) || *cptr == '.');
			++i;
		}
	} else if(*cptr == '+')
	{
		++i;
		update_expression(ADD);
	} else if(*cptr == '-')
	{
		++i;
		update_expression(SUBSTRACT);
	} else if(*cptr == '*')
	{
		++i;
		update_expression(MULTIPLY);
	} else if(*cptr == '/')
	{
		++i;
		update_expression(DIVIDE);
	} else if(*cptr == '|')
	{
		update_expression(ABS);
	} else if(is_equal_str(cptr, "exp"))
	{
		i += 3;
		update_expression(EXP);
	} else if(is_equal_str(cptr, "ln"))
	{
		i += 2;
		update_expression(LN);
	} else if(is_equal_str(cptr, "pow"))
	{
		i += 3;
		update_expression(POW);
	} else if(is_equal_str(cptr, "cos"))
	{
		i += 3;
		update_expression(COS);
	} else if(is_equal_str(cptr, "sin"))
	{
		i += 3;
		update_expression(SIN);
	} else if(is_equal_str(cptr, "tan"))
	{
		i += 3;
		update_expression(TAN);
	} else if(is_equal_str(cptr, "sec"))
	{
		i += 3;
		update_expression(SEC);
	} else if(is_equal_str(cptr, "cosec"))
	{
		i += 4;
		update_expression(COSEC);
	} else if(is_equal_str(cptr, "cot"))
	{
		i += 3;
		update_expression(COT);
	} else if(is_equal_str(cptr, "acos"))
	{
		i += 4;
		update_expression(ACOS);
	} else if(is_equal_str(cptr, "asin"))
	{
		i += 4;
		update_expression(ASIN);
	} else if(is_equal_str(cptr, "atan"))
	{
		i += 4;
		update_expression(ATAN);
	} else if(is_equal_str(cptr, "asec"))
	{
		i += 4;
		update_expression(ASEC);
	} else if(is_equal_str(cptr, "acosec"))
	{
		i += 6;
		update_expression(ACOSEC);
	} else if(is_equal_str(cptr, "acot"))
	{
		i += 4;
		update_expression(ACOT);
	} else if(*cptr == '(')
	{
		++i;
		update_expression(BRACKET);
	} else if(*cptr == ')')
	{
		++i;
		update_expression(BRACKET);
	} else if(*cptr == ' ')
	{
		++i;
	} else
	{
		printf("Give Valid Expression\n");
		exit(EXIT_FAILURE);
	}
	return i;
}

//preliminary idea on the given expression consisting of an array of operators in user given sequence
void get_expression(char *input)
{
	int i = 0;
	while(input[i] != '\0')
	{
		i += get_operator_and_update_index(&input[i]);
	}
	update_expression(EXIT);
	return;
}

void gen_reverse_polish_notation(enum operator_ *);

//in which sequence the operations should be done
void gen_operator_sequence(char *input)
{
	extern enum operator_ *expression;
	get_expression(input);
	gen_reverse_polish_notation(expression);
	return;
}

//executing single variable function. either a real type or a complex type depending on the value of number_system
void exec_svar_func(double (*funcr)(double), complex double (*funcc)(complex double))
{
	extern enum number_system_ number_system;
	extern struct calc_stack_ calc_stack;
	extern int offset;

	void *numptr = malloc((size_t)offset);
	switch(number_system)
	{
		case real_ :
			*((double *)numptr) = funcr(((double *)calc_stack.ptr)[calc_stack.index]);
			break;
		case complex_ :
			*((complex double*)numptr) = funcc(((complex double*)calc_stack.ptr)[calc_stack.index]);
			break;
	}
	calc_stack.pop();
	calc_stack.push(numptr);
	free(numptr);
	return;
}

//executing double variable function. either a real type or a complex type depending on the value of number_system
void exec_dvar_func(double (*funcr)(double, double), complex double (*funcc)(complex double, complex double))
{
	extern enum number_system_ number_system;
	extern int offset;
	extern struct calc_stack_ calc_stack;

	void *numptr = malloc((size_t)offset);
	switch(number_system)
	{
		case real_ :
			*((double *)numptr) = funcr(((double *)calc_stack.ptr)[calc_stack.index], ((double *)calc_stack.ptr)[calc_stack.index - 1]);
			break;
		case complex_ :
			*((complex double *)numptr) = funcc(((complex double *)calc_stack.ptr)[calc_stack.index], ((complex double *)calc_stack.ptr)[calc_stack.index - 1]);
			break;
	}
	calc_stack.pop();
	calc_stack.pop();
	calc_stack.push(numptr);
	free(numptr);
	return;
}

//user defined math functions
double add(double num1, double num2){return num1 + num2;}
double substract(double num1, double num2){return num1 - num2;}
double multiply(double num1, double num2){return num1*num2;}
double divide(double num1, double num2){return num1/num2;}
double sec(double num){return 1/cos(num);}
double cosec(double num){return 1/sin(num);}
double cot(double num){return 1/tan(num);}
double asec(double num){return acos(1/num);}
double acosec(double num){return asin(1/num);}
double acot(double num){return atan(1/num);}
complex double cadd(complex double num1, complex double num2){return num1 + num2;}
complex double csubstract(complex double num1, complex double num2){return num1 - num2;}
complex double cmultiply(complex double num1, complex double num2){return num1*num2;}
complex double cdivide(complex double num1, complex double num2){return num1/num2;}
complex double csec(complex double num){return 1/ccos(num);}
complex double ccosec(complex double num){return 1/csin(num);}
complex double ccot(complex double num){return 1/ctan(num);}
complex double casec(complex double num){return cacos(1/num);}
complex double cacosec(complex double num){return casin(1/num);}
complex double cacot(complex double num){return catan(1/num);}
complex double cabs_(complex double num){return (complex double)(cabs(num));}

//execute a certain operation
void operate(enum operator_ operator)
{
	extern int offset;
	extern struct calc_stack_ calc_stack;
	extern struct num_stack_ num_stack;
	switch(operator)
	{
		case NUMBER :
			calc_stack.push((void *)((char *)num_stack.ptr + offset*num_stack.index++));
			break;
		case ADD :
			exec_dvar_func(add, cadd);
			break;
		case SUBSTRACT :
			exec_dvar_func(substract, csubstract);
			break;
		case MULTIPLY :
			exec_dvar_func(multiply, cmultiply);
			break;
		case DIVIDE :
			exec_dvar_func(divide, cdivide);
			break;
		case ABS :
			exec_svar_func(fabs, cabs_);
		case EXP :
			exec_svar_func(exp, cexp);
			break;
		case LN :
			exec_svar_func(log, clog);
			break;
		case POW :
			exec_dvar_func(pow, cpow);
			break;
		case COS :
			exec_svar_func(cos, ccos);
			break;
		case SIN :
			exec_svar_func(sin, csin);
			break;
		case TAN :
			exec_svar_func(tan, ctan);
			break;
		case SEC :
			exec_svar_func(sec, csec);
			break;
		case COSEC :
			exec_svar_func(cosec, ccosec);
			break;
		case COT :
			exec_svar_func(cot, ccot);
			break;
		case ACOS :
			exec_svar_func(acos, cacos);
			break;
		case ASIN :
			exec_svar_func(asin, casin);
			break;
		case ATAN :
			exec_svar_func(atan, catan);
			break;
		case ASEC :
			exec_svar_func(asec, casec);
			break;
		case ACOSEC :
			exec_svar_func(acosec, cacosec);
			break;
		case ACOT :
			exec_svar_func(acot, cacot);
			break;
		case BRACKET :
			break;
		case EXIT :
			break;
	}
}

void gen_result(void)
{
	extern enum number_system_ number_system;
	extern enum operator_ *operator;
	extern struct calc_stack_ calc_stack;
	for(int i = 0; operator[i] != EXIT; ++i)
	{
		operate(operator[i]);
	}

	switch(number_system)
	{
		case real_ :
			printf("Ans = %lf\n", *((double *)calc_stack.ptr));
			break;
		case complex_ :
			printf("Ans = %lf + %lf i\n", creal(*((complex double *)calc_stack.ptr)), cimag(*((complex double *)calc_stack.ptr)));
			break;
	}
	return;
}
