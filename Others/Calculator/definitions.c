#ifndef STACKS
#define STACKS

//for storing which operation to do
struct calc_stack_
{
	void *ptr;
	int index;
	void (*push)(void *);
	void (*pop)(void);
};

//for storing numbers only
struct num_stack_
{
	void *ptr;
	int index;
	void (*push)(void *);
};

#endif

#ifndef ENUMS
#define ENUMS

enum number_system_ {real_, complex_};

enum operator_
{
	NUMBER,
	ADD,
	SUBSTRACT,
	MULTIPLY,
	DIVIDE,
	ABS,
	EXP,
	LN,
	POW,
	COS,
	SIN,
	TAN,
	SEC,
	COSEC,
	COT,
	ACOS,
	ASIN,
	ATAN,
	ASEC,
	ACOSEC,
	ACOT,
	BRACKET,
	EXIT
};

#endif
