#include "counter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Se debe ingresar el tamaño de la matriz\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int max_iter = 20000;
    float tol = 1e-8;
    float tiempo = -1;

    int iter = 0;
    srand(n);

    float **a = (float **)aligned_alloc(64, n * sizeof(float *));
    for (int i = 0; i < n; i++)
    {
        a[i] = (float *)aligned_alloc(64, n * sizeof(float));
    }
    float *x_new = (float *)aligned_alloc(64, n * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        float row_sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            a[i][j] = (float)rand() / RAND_MAX;
            row_sum += fabs(a[i][j]);
        }
        a[i][i] = row_sum + 1.0f;
    }

    float *b = (float *)aligned_alloc(64, n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        b[i] = (float)rand() / RAND_MAX;
    }

    float *x = (float *)aligned_alloc(64, n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
    }

    float norm2;
    int bsize = 2000; // Tamaño de bloque ajustable

    start_counter();

    for (iter = 0; iter < max_iter; iter++)
    {
        norm2 = 0.0f;

        for (int i = 0; i < n; i++)
        {
            float sigma = 0.0f;

            // Recorrer la matriz en bloques para mejorar la localidad en caché
            for (int jj = 0; jj < n; jj += bsize) // Bloques de columnas
            {
                for (int j = jj; j < jj + bsize && j < n; j++) // Acceso dentro del bloque
                {
                    if (i != j)
                    {
                        sigma += a[i][j] * x[j];
                    }
                }
            }

            x_new[i] = (b[i] - sigma) / a[i][i];
            norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
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
    printf("N: %d. Tiempo: %.10f (~%.2f segundos)\n", n,tiempo, tiempo/10e8);
    printf("Norm\u00b2: %.14f\n", norm2);

    free(a);
    free(b);
    free(x);
    free(x_new);
}
