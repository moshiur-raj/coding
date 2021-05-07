#include "definitions.c"
#include <stdlib.h>

void gen_reverse_polish_notation(enum operator_ *expression)
{
	extern enum operator_ *operator;

	//temp solution
	#include <string.h>
	extern int expression_index;
	operator = (enum operator_ *)malloc((expression_index + 1)*sizeof(enum operator_));
	memcpy((void *)operator, (void *)expression, (expression_index + 1)*sizeof(enum operator_));

	return;
}
