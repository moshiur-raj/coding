#include <stdio.h>
#include <quadmath.h>

#define k		273.15q
#define m		0.02897q
#define R		8.31446261815324q
#define g0		9.807q
#define kg		(2*9.807q/6378.1e3q)
#define step1	1e-9q
#define step2	1
#define d0		1.225q
#define p0		101300

int main()
{
	__float128 h[11] = {0, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3};
	__float128 T[11] = {15 + k, 8.5q + k, 2 + k, -4.49q + k, -10.98q + k, -17.47q + k, -23.96q + k, -30.45q + k, -36.94q + k, -43.42q + k, -49.90q + k};

	__float128 e = 0, e_min = 1e9, kt_opt, kt = 0;
	while (kt <= 1e-2q)
	{
		for (int i = 0; i <= 10; ++i)
		{
			e = e + fabsq(T[0] - kt*h[i] - T[i])/T[i];
		}
		if (e < e_min)
		{
			e_min = e;
			kt_opt = kt;
		}
		e = 0;
		kt = kt + step1;
	}

	__float128 A, B, C;
	A = m*kg/kt_opt/R;
	B = g0/kg - kt_opt*R/m/kg;
	C = T[0]/kt_opt;

	long long int n = h[10]/step2;
	__float128 H[n + 1], d[n + 1], p[n + 1];
	for (long long int i = 0; i <= n; ++i)
	{
		H[i] = i*step2;
		d[i] = d0*expq(-A*(H[i] + (C - B)*logq(fabsq(1 - H[i]/C))));
		p[i] = d[i]*R*(T[0] - kt_opt*H[i])/m;

		printf("%Qf\n %.6Qf\n %.3Qf\n", H[i], d[i], p[i]);
	}

	printf("end\n");
	printf("%Qf\n %Qf\n %Qf\n %Qf\n %Qf\n", kt_opt, kg, A, B, C);

	return 0;
}