#include "counter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>
#include <stdalign.h>

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
    float **a = (float **)aligned_alloc(32, n * sizeof(float *));
    for (int i = 0; i < n; i++)
    {
        a[i] = (float *)aligned_alloc(32, n * sizeof(float));
    }

    float *x_new = (float *)aligned_alloc(32, n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        float row_sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            a[i][j] = (float)rand() / RAND_MAX;
            row_sum += a[i][j];
        }
        a[i][i] += row_sum;
    }

    float *b = (float *)aligned_alloc(32, n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        b[i] = (float)rand() / RAND_MAX;
    }

    float *x = (float *)aligned_alloc(32, n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
    }
    alignas(32) float diagonal[n];

    int j;
    float norm2;
    __m256 av, xv, cv;

    start_counter();
    for (iter = 0; iter < max_iter; iter++)
    {
        norm2 = 0.0f;
        for (int i = 0; i < n; i++)
        {
            if (iter == 0)
            {
                diagonal[i] = a[i][i];
                a[i][i] = 0.0f;
            }
            float sigma = 0.0f;
            for (j = 0; j < n - 8; j += 8)
            {

                av = _mm256_load_ps(&a[i][j]);
                xv = _mm256_load_ps(&x[j]);
                cv = _mm256_mul_ps(av, xv);

                alignas(32) float vector_temp[8];
                _mm256_store_ps(vector_temp, cv);
                for (int k = 0; k < 8; k++)
                {
                    sigma += vector_temp[k];
                }
            }
            for (int l = j; l < n; l++)
            {
                if (i != l)
                {
                    sigma += a[i][l] * x[l];
                }
            }

            x_new[i] = (b[i] - sigma) / diagonal[i];
            norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        for (int i = 0; i < n; i++)
        {
            x[i] = x_new[i];
        }
        if (sqrt(norm2) < tol){
            tiempo = get_counter();
            break;
        }
    }
    

    if (tiempo == -1)
    {
        tiempo = get_counter();
    }

    printf("Norma final: %e\n", sqrt(norm2));
    printf("Número de ciclos: %.2lf\n", tiempo);

    exit(0);
}
