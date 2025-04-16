#include "counter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

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

    int j;
    __m256 av, xv, cv;
    start_counter();
    for (iter = 0; iter < max_iter; iter++)
    {
        float norm2 = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float sigma = 0.0f;
            for (j = 0; j < n - 8; j += 8)
            {
                _Alignas(32) int vector_mask[8];
                for (int k = 0; k < 8; k++)
                {
                    if (j + k == i)
                    {
                        vector_mask[k] = 69; // bit mas alto --> 0 --> no leer
                    }
                    else
                        vector_mask[k] = -69; // bit mas alto --> 1 --> leer
                }
                __m256i mask = _mm256_setr_epi32(vector_mask[0], vector_mask[1], vector_mask[2], vector_mask[3], vector_mask[4], vector_mask[5], vector_mask[6], vector_mask[7]);
                av = _mm256_maskload_ps(&a[i][j], mask);
                
                if (i - 8 <= j <= i)
                {
                    printf("i: %d, j: %d, vector_mask: [", i, j);
                    for (int k = 0; k < 8; k++)
                    {
                        printf("%d", vector_mask[k]);
                        if (k < 7)
                            printf(", ");
                    }
                    printf("]\n");

                    _Alignas(32) int mask_array[8];
                    _mm256_store_si256((__m256i *)mask_array, mask);
                    printf("Mask: [");
                    for (int k = 0; k < 8; k++)
                    {
                        printf("%d", mask_array[k]);
                        if (k < 7)
                            printf(", ");
                    }
                    printf("]\n");

                    _Alignas(32) float av_array[8];
                    _mm256_store_ps(av_array, av);
                    printf("av: [");
                    for (int k = 0; k < 8; k++)
                    {
                        printf("%f", av_array[k]);
                        if (k < 7)
                            printf(", ");
                    }
                    printf("]\n");
                }
                xv = _mm256_load_ps(&x[j]);
                cv = _mm256_mul_ps(av, xv);
                _Alignas(32) float vector_temp[8];
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

            x_new[i] = (b[i] - sigma) / a[i][i];
            norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        if (sqrt(norm2) < tol)
        {
            printf("Iterations: %d\n", iter);
            tiempo = get_counter();
            printf("Norm: %f\n", sqrt(norm2));
            printf("Time: %f\n", tiempo);
            exit(0);
        }
    }
    tiempo = get_counter();
    printf("Time: %f\n", tiempo);
    exit(0);
}
