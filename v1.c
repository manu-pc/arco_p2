#include "counter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
    float tiempo = -1;
    if (argc < 2)
    {
        printf("Se debe ingresar el tamaño de la matriz\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int max_iter = 20000;
    float tol = 1e-8;
    int iter = 0;
    srand(n);

    float **a = malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++)
    {
        a[i] = malloc(n * sizeof(float));
    }

    for (int i = 0; i < n; i++)
    {
        float row_sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            a[i][j] = (float)rand() / RAND_MAX;
            row_sum += fabs(a[i][j]);
        }
        a[i][i] = row_sum + 1.0; // diagonal dominante
        // Lo es por filas cuando, para todas las filas, el valor absoluto del elemento de la diagonal de esa fila es estrictamente mayor que la norma del resto de elementos de esa fila.
    }

    float *b = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        b[i] = (float)rand() / RAND_MAX;
    }

    float *x = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
    }
    float *x_new = malloc(n * sizeof(float));
    float norm2;
    start_counter();

    for (iter = 0; iter < max_iter; iter++)
    {
        norm2 = 0.0;
        for (int i = 0; i < n; i++)
        {
            float sigma = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sigma += a[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / a[i][i];
            norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }

        // x = x_new
        for (int i = 0; i < n; i++)
        {
            x[i] = x_new[i];
        }
        if (sqrt(norm2) < tol)
        {
            tiempo = get_counter();
            printf("Converge en %d iteraciones.\n", iter);
            break;
        }
    }

    if (tiempo == -1)
    {
        tiempo = get_counter();
        printf("Se ha alcanzado el máximo de iteraciones (%d) sin llegar a la solución.\n", iter);
    }
    printf("N: %d. Tiempo: %.10f (~%.2f segundos)\n",n, tiempo, tiempo / 10e8);
    printf("Norm\u00b2: %.14f\n", norm2);
    free(x_new);

    for (int i = 0; i < n; i++)
    {
        free(a[i]);
    }
    free(a);
    free(b);
    free(x);
}
