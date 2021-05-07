#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern double Alpha;
extern double Length;
extern double TimeMin;
extern double TimeMax;
extern int NumPositions;
extern int Nmax;
extern int NumFrames;

extern inline double initial_temp(double);

static double *Position,
			  *Time,
			  *Temperature,
			  *CoeffFourier,
			  *CoeffCos,
			  *CoeffExp;


static double PositionStep,
			  TimeStep;

static void alloc_global_vars(void)
{
	Position = malloc(sizeof(double)*NumPositions);
	Time = malloc(sizeof(double)*NumFrames);
	Temperature = malloc(sizeof(double)*NumPositions);
	CoeffFourier = malloc(sizeof(double)*(Nmax + 1));
	CoeffCos = malloc(sizeof(double)*(Nmax + 1));
	CoeffExp = malloc(sizeof(double)*(Nmax + 1));
}

static void gen_time_and_position(void)
{
	PositionStep = 1.0l*Length/(NumPositions - 1);
	for(int i = 0; i < NumPositions - 1; ++i)
	{
		Position[i] = 1.0l*i*PositionStep;
	}
	Position[NumPositions - 1] = Length;

	TimeStep = 1.0l*(TimeMax - TimeMin)/NumFrames;
	for(int i = 0; i < NumFrames - 1; ++i)
	{
		Time[i] = TimeMin + i*TimeStep;
	}
	Time[NumFrames - 1] = TimeMax;
}

static inline double integrate(double (*function)(double),
								double coeffFunction)
{
	double integral = 0;
	for(int i = 0; i < NumPositions; ++i)
	{
		integral += initial_temp(Position[i])*function(coeffFunction*Position[i])*PositionStep;
	}
	return integral;
}

static void calculate_coefficients(void)
{
	double temporaryConstant;
	for(int i = 0; i <= Nmax; ++i)
	{
		temporaryConstant = i*M_PI/Length;
		CoeffFourier[i] = 2/Length*integrate(cos, temporaryConstant);
		CoeffCos[i] = temporaryConstant;
		CoeffExp[i] = -1*Alpha*temporaryConstant*temporaryConstant;
	}
}

static inline void print_array(double *array, int arraySize)
{
	for(int i = 0; i < arraySize; ++i)
	{
		printf("%le\t", array[i]);
	}
	printf("\n");
}

static inline void print_initial_temps(void)
{
	for(int i = 0; i < NumPositions; ++i)
	{
		printf("%le\t", initial_temp(Position[i]));
	}
	printf("\n");
}

static inline double fourier_series(double position, double time)
{
	double fourierSum = CoeffFourier[0]/2;
	for(int i = 1; i <= Nmax; ++i)
	{
		fourierSum += CoeffFourier[i]*exp(CoeffExp[i]*time)*cos(CoeffCos[i]*position);
	}
	return fourierSum;
}

static inline void calculate_temps(int index)
{
	for(int i = 0; i < NumPositions; ++i)
	{
		Temperature[i] = fourier_series(Position[i], Time[index]);
	}
}

static void gen_data(void)
{
	print_initial_temps();
	for(int i = 0; i < NumFrames; ++i)
	{
		calculate_temps(i);
		print_array(Temperature, NumPositions);
	}
}

void start_calculation(void)
{
	alloc_global_vars();
	gen_time_and_position();
	calculate_coefficients();

	print_array(Position, NumPositions);
	gen_data();
}
