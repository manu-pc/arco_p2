#include "counter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Se debe ingresar el tama\u00f1o de la matriz\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int max_iter = 20000;
    float tol = 1e-8;
    int iter = 0;
    float tiempo = -1;
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
    _Alignas(32) float sigma_vec[8];
    __m256 av, bv, cv;
    start_counter();
    for (iter = 0; iter < max_iter; iter++)
    {
        norm2 = 0.0;
        for (int i = 0; i < n; i += 1)
        {
            float sigma = 0.0;
            for (int j = 0; j < n; j += 8)
            {
                if (i >= j && i <= (j + 7)) continue;
                av = _mm256_load_ps(a[i] + j);
                bv = _mm256_load_ps(x + j);
                cv = _mm256_mul_ps(av, bv);
                _mm256_store_ps(sigma_vec, cv);
                sigma += (sigma_vec[0] + sigma_vec[1] + sigma_vec[2] + sigma_vec[3] +
                          sigma_vec[4] + sigma_vec[5] + sigma_vec[6] + sigma_vec[7]);
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

    if (tiempo == -1){
        tiempo = get_counter();
        printf("Se ha alcanzado el máximo de iteraciones (%d) sin llegar a la solución.\n", iter);
    }
    printf("N: %d. Tiempo: %.10f (~%.2f segundos)\n", n, tiempo, tiempo / 10e8);
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