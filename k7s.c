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

    int j;
    float norm2;
    _Alignas(32) float sigma_vec[8];
    __m256 av, bv, cv;
     __m256 maskfloat;
    __m256 zero = _mm256_setzero_ps();
    __m256i mascara;
    start_counter();
    for (iter = 0; iter < max_iter; iter++)
    {
        norm2 = 0.0;
        for (int i = 0; i < n; i += 1)
        {

            float sigma = 0.0;
            __m256 sigma_reg = _mm256_setzero_ps();

            for (j = 0; j < n - 8; j += 8)
            {

                    _Alignas(32) float vector[8];
                    for (int k = 0; k < 8; k++)
                    {
                        if (j + k == i)
                            vector[k] = 0;
                        else
                        {
                            // printf("i: %d, j: %d, k: %d, j+k: %d\n", i, j, k, j + k);
                            vector[k] = -1;
                        }
                    }
                     maskfloat = _mm256_load_ps(vector);
                    
                    // printf("i: %d, j: %d, vector: [", i, j);
                    // for (int k = 0; k < 8; k++)
                    // {
                    //     printf("%f", vector[k]);
                    //     if (k < 7)
                    //         printf(", ");
                    // }
                    // printf("]\n");

                

                //? INTENTO 1: GITANADA
                // av = _mm256_load_ps(a[i] + j);
                // if (j <= i && j >= (i - 7)) continue;

                //? INTENTO 2: MASKLOAD


                av = _mm256_load_ps(a[i] + j);

                bv = _mm256_load_ps(&x[j]);
                cv = _mm256_mul_ps(av, bv);
                
                //? INTENTO 3: BLEND

                //! perú es clave
                __m256 ms2 = _mm256_blendv_ps(zero, cv, maskfloat);
                sigma_reg = _mm256_add_ps(sigma_reg, ms2);

                // _Alignas(32) float maskfloat_vals[8];
                // _mm256_store_ps(maskfloat_vals, maskfloat);
                // printf("maskfloat = [");
                // for (int k = 0; k < 8; k++)
                // {
                //     printf("%.6f", maskfloat_vals[k]);
                //     if (k < 7)
                //         printf(", ");
                // }
                // printf("]\n");

                // if (j <= i && j >= (i - 7))
                // {
                //     _Alignas(32) float vals[8];
                //     _mm256_store_ps(vals, mascara);
                //     printf("i = %d, j = %d\n", i, j);
                //     printf("mascara = [");
                //     for (int i = 0; i < 8; i++)
                //     {
                //         printf("%f", vals[i]);
                //         if (i < 7)
                //             printf(", ");
                //     }
                //     printf("]\n");}
                //     _Alignas(32) float av_vals[8];
                //     _mm256_store_ps(av_vals, av);
                //     printf("av = [");
                //     for (int i = 0; i < 8; i++)
                //     {
                //         printf("%.6f", av_vals[i]);
                //         if (i < 7)
                //             printf(", ");
                //     }
                //     printf("]\n");
                //     _Alignas(32) float cv_vals[8];
                //     _mm256_store_ps(cv_vals, cv);
                //     printf("cv = [");
                //     for (int i = 0; i < 8; i++)
                //     {
                //         printf("%.6f", cv_vals[i]);
                //         if (i < 7)
                //             printf(", ");
                //     }
                //     printf("]\n");
                // }

                // _mm256_store_ps(sigma_vec, cv);
                // sigma += (sigma_vec[0] + sigma_vec[1] + sigma_vec[2] + sigma_vec[3] +
                //           sigma_vec[4] + sigma_vec[5] + sigma_vec[6] + sigma_vec[7]);
            

            }
            for (int l = j; l < n; l++)
            {
                if (i != l)
                {
                    sigma += (a[i][l]) * x[l];
                }
            }
            if (fabs(a[i][i]) < 1e-10)
            {
                printf("Elemento diagonal muy pequeño o nulo en fila %d: %f\n", i, a[i][i]);
                exit(1);
            }

            _mm256_store_ps(sigma_vec, sigma_reg);
            sigma += (sigma_vec[0] + sigma_vec[1] + sigma_vec[2] + sigma_vec[3] +
                      sigma_vec[4] + sigma_vec[5] + sigma_vec[6] + sigma_vec[7]);

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
            //printf("iter: %d, norm2: %.14f, sqrt(norm2): %.14f, tol: %.14f\n", iter, norm2, sqrt(norm2), tol);

            printf("Converge en %d iteraciones.\n", iter);
            break;
        }

        if (iter < max_iter)
        {
            printf("iter: %d, norm2: %.14f, sqrt(norm2): %.14f, tol: %.14f\n", iter, norm2, sqrt(norm2), tol);
        }
    }

    if (tiempo == -1)
    {
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