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
            row_sum += a[i][j];
        }
        a[i][i] += row_sum; // diagonal dominante
        // Lo es por filas cuando, para todas las filas, el valor absoluto del elemento de la diagonal de esa fila es estrictamente mayor que la norma del resto de elementos de esa fila.
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
    int bsize = 64; // Tamaño de bloque ajustable

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
                    sigma += a[i][j] * x[j];
            }

            // sigma -= a[i][i]*x[i];
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
        //         else {
        //     if (iter < max_iter){
        //         printf("iter: %d, norm2: %.14f, sqrt(norm2): %.14f, tol: %.14f\n", iter, norm2, sqrt(norm2), tol);
        //     }
        // }
    }

    if (tiempo == -1)
    {
        tiempo = get_counter();
        printf("Se ha alcanzado el máximo de iteraciones (%d) sin llegar a la solución.\n", iter);
    }
    printf("N: %d. Tiempo: %.10f (~%.2f segundos)\n", n, tiempo, tiempo / 10e8);
    printf("Norm\u00b2: %.14f\n", norm2);

    free(a);
    free(b);
    free(x);
    free(x_new);
}
