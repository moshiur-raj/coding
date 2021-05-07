#include <stdio.h>
#include <quadmath.h>

#define G	6.67430e-11q	//gravitational constant in SI units
#define R	6371e3			//radius of the earth in meters
#define dt	1e-3			//time step in seconds

#define number_of_layers					10		//number of layers in earth
#define degree_of_density_polynomial		3		//degree of the polynomial used to approximate density in a layer

/*Inner radius of the layers in meters*/
#define inner_core_inner_radius				0
#define outer_core_inner_radius				1221.5e3
#define lower_mantle_inner_radius			3480e3
#define	transition_zone_1_inner_radius		5701e3
#define	transition_zone_2_inner_radius		5771e3
#define	transition_zone_3_inner_radius		5971e3
#define	lvz_and_lid_inner_radius			6151e3
#define crust_1_inner_radius				6346.6e3
#define crust_2_inner_radius				6356e3
#define ocean_inner_radius					6368e3
#define space_inner_radius					6371e3

/*Co-efficients of the polynomials used to approximate density for a layer in kg/m^3. Co-efficient of Lower order term first (from power of 0 to 3).The polynomial takes normalized radius (r/R) as input*/
#define density_polynomial_inner_core			{13.0885e3q, 0, -8.8381e3q, 0}						//for radius in      0-1221.5km
#define density_polynomial_outer_core			{12.5815e3q, -1.2638e3q, -3.6426e3q, -5.5281e3q}	//for radius in 1221.5-3480.0km
#define density_polynomial_lower_mantle			{7.9565e3q, -6.4761e3q, 5.5283e3q, -3.0807e3q}		//for radius in 3840.0-5701.0km
#define density_polynomial_transition_zone_1	{5.3197e3q, -1.4836e3q, 0, 0}						//for radius in 5701.0-5771.0km
#define density_polynomial_transition_zone_2	{11.2494e3q, -8.0298e3q, 0, 0}						//for radius in 5771.0-5971.0km
#define density_polynomial_transition_zone_3	{7.1089e3q, -3.8045e3q, 0, 0}						//for radius in 5791.0-6151.0km
#define density_polynomial_lvz_and_lid			{2.6910e3q, 0.6924e3q, 0, 0}						//for radius in 6151.0-6346.6km
#define density_polynomial_crust_1				{2.900e3q, 0 , 0, 0}								//for radius in 6346.6-6356.0km
#define density_polynomial_crust_2				{2.600e3q, 0, 0, 0}									//for radius in 6356.0-6368.0km
#define density_polynomial_ocean				{1.020e3q, 0, 0, 0}									//for radius in 6368.0-6371.0km

/*Coefficients of the polynomial acquired after integrating 4*pi*r^2*density(r) with respect to r*/
#define coefficient_for_determining_mass	{1/3.0q, 1/(4.0q*R), 1/(5.0q)*powq(R, -2), 1/(6.0q)*powq(R, -3)}

/*For plotting*/
#define dr							1e-6q						//(R_max - R_min)/dr > number_of_points_to_plot
#define R_min						dr*1e3
#define R_max						(2*space_inner_radius)
#define number_of_points_to_plot	(int)1e4

int main()
{
	__float128 density_polynomial_array[number_of_layers][degree_of_density_polynomial + 1] = {density_polynomial_inner_core, density_polynomial_outer_core, density_polynomial_lower_mantle, density_polynomial_transition_zone_1, density_polynomial_transition_zone_2, density_polynomial_transition_zone_3, density_polynomial_lvz_and_lid, density_polynomial_crust_1, density_polynomial_crust_2, density_polynomial_ocean};
	__float128 inner_radius_array[number_of_layers + 1] = {inner_core_inner_radius, outer_core_inner_radius, lower_mantle_inner_radius, transition_zone_1_inner_radius, transition_zone_2_inner_radius, transition_zone_3_inner_radius, lvz_and_lid_inner_radius, crust_1_inner_radius, crust_2_inner_radius, ocean_inner_radius, space_inner_radius};
	__float128 coefficient[degree_of_density_polynomial + 1] = coefficient_for_determining_mass;

	/*Determining mass of each layer*/
	__float128 mass_array[number_of_layers];
	int i, j;
	for (i = 0; i < number_of_layers; ++i)
	{
		mass_array[i] = 0;
		for (j = 0; j < degree_of_density_polynomial + 1; ++j)
		{
			mass_array[i] = mass_array[i] + coefficient[j]*density_polynomial_array[i][j]*(powq(inner_radius_array[i + 1], j + 3) - powq(inner_radius_array[i], j + 3));
		}
		mass_array[i] = mass_array[i]*4*M_PIq;
	}

	/*Mass of the ball encompassed by radius <= x where x<= space_inner_radius*/
	__float128 m;
	__float128 mass(__float128 x)
	{
		m = 0;
		for (i = 0; i < number_of_layers ; ++i)			//Counting number of layers before radius = x using i
		{
			if (x > inner_radius_array[i + 1])
			{
				m = m + mass_array[i];
			}
			else
			{
				break;
			}
		}
		return m + (coefficient[0]*density_polynomial_array[i][0]*(x*x*x - inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]) + coefficient[1]*density_polynomial_array[i][1]*(x*x*x*x - inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]) + coefficient[2]*density_polynomial_array[i][2]*(x*x*x*x*x - inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]) + coefficient[3]*density_polynomial_array[i][3]*(x*x*x*x*x*x - inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]*inner_radius_array[i]))*4*M_PIq;
	}

	__float128 v = 0, r = 6371e3, r0, g, k = 0;
	while (r > 0)
	{
		g = G*mass(r)*powq(r , -2);
		r0 = r;
		r = r + v*dt - g*dt*dt/2;
		v = v - g*dt + g/r0*v*dt*dt;
		++k;
	}
	printf("The required time is %.6Qfs\n", k*dt);

	__float128 M = mass(inner_radius_array[number_of_layers]), V = 0, p;
	r = R_min;
	j = 0;
	k = 1;
	if (R_min <= space_inner_radius)
	{
		while (r <= space_inner_radius)
		{
			m = mass(r);
			g = G*m/(r*r);
			if (k >= j*(R_max - R_min)/(dr*number_of_points_to_plot))
			{
				printf("%.6Qf\n", r);
				printf("%.6Qf\n", m);
				printf("%.6Qf\n", V);
				printf("%.6Qf\n", g);
				++j;
			}
			V = V + g*dr;
			r = R_min + k*dr;
			++k;
		}
		while (r <= R_max)
		{
			g = G*M/(r*r);
			if (k >= j*(R_max - R_min)/(dr*number_of_points_to_plot))
			{
				printf("%.6Qf\n", r);
				printf("%.6Qf\n", M);
				printf("%.6Qf\n", V);
				printf("%.6Qf\n", g);
				++j;
			}
			V = V + g*dr;
			r = R_min + k*dr;
			++k;
		}
		printf("%.6Qf\n", r);
		printf("%.6Qf\n", M);
		printf("%.6Qf\n", V);
		printf("%.6Qf\n", G*M/(r*r));
	}
	else
	{
		while (r <= R_max)
		{
			g = G*M/(r*r);
			if (k >= j*(R_max - R_min)/(dr*number_of_points_to_plot))
			{
				printf("%.6Qf\n", r);
				printf("%.6Qf\n", M);
				printf("%.6Qf\n", V);
				printf("%.6Qf\n", g);
				++j;
			}
			V = V + g*dr;
			r = R_min + k*dr;
			++k;
		}
		printf("%.6Qf\n", r);
		printf("%.6Qf\n", M);
		printf("%.6Qf\n", V);
		printf("%.6Qf\n", G*M/(r*r));
	}
	printf("end\n");

	return 0;
}