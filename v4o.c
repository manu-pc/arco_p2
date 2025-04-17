#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "counter.h"

#define k 8

int main(int argc, char *argv[]) {
    float tiempo = -1;
    if (argc < 2) {
        printf("Uso: %s <N> <t>\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
	int t = atoi(argv[2]);
    float **a = (float **)aligned_alloc(64, N * sizeof(float *));
    if (!a) {
        printf("Error al asignar memoria para la matriz.\n");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        a[i] = (float *)aligned_alloc(64, N * sizeof(float));
        if (!a[i]) {
            printf("Error al asignar memoria para la fila %d de la matriz.\n", i);
            return 1;
        }
    }
    
    float *b = (float *)aligned_alloc(64, N * sizeof(float));
    if (!b) {
        printf("Error al asignar memoria para el vector.\n");
        return 1;
    }
    float *x = (float *)aligned_alloc(64, N * sizeof(float));
    if (!x) {
        printf("Error al asignar memoria para la solución.\n");
        return 1;
    }
    
    float tol = 1e-8;
    int max_iter = 20000;
    
    srand(N);
    for (int i = 0; i < N; i++) {
        float suma_fila = 0;
        for (int j = 0; j < N; j++) {
            a[i][j] = (float)rand() / RAND_MAX;
            suma_fila += a[i][j];
        }
        a[i][i] += suma_fila;
        b[i] = (float)rand() / RAND_MAX;
        x[i] = 0.0;
    }
    
    float *x_new = (float *)aligned_alloc(64, N * sizeof(float));
    if (!x_new) {
        printf("Error al asignar memoria.\n");
        return 1;
    }
    
    start_counter();
    int iter = 0;
    float norm2;
    for (iter = 0; iter < max_iter; iter++) {
        norm2 = 0.0;
       
 int j;
        #pragma omp parallel private(j) reduction(+:norm2)  num_threads(t)
        {
        #pragma omp for
            for (int i = 0; i < N; i++) {
                double sigma = 0.0;
                
                for (j = 0; j < N; j++) {
                    if (i != j) {
                        sigma += a[i][j] * x[j];
                    }
                }
                
                x_new[i] = (b[i] - sigma) / a[i][i];
                norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
                x[i] = x_new[i];
            }
        }
        
        
    
        /* Este trozo hace sections y va m as rapido
        #pragma omp parallel private(j) reduction(+:norm2) num_threads(t)
        {
                #pragma omp sections
                {
                    // Sección 1: Calcular sigma y actualizar x_new
                    #pragma omp sections
                    {
                        for (int i = 0; i < N; i++) {
                            double sigma = 0.0;

                            // Calcular sigma
                            for (j = 0; j < N; j++) {
                                if (i != j) {
                                    sigma += a[i][j] * x[j];
                                }
                            }

                            x_new[i] = (b[i] - sigma) / a[i][i];
                            norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
                            x[i] = x_new[i];

                        }
                    }
                }
            
        }*/
        
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
    printf("Tiempo: %.10f (~%.2f segundos)\n", tiempo, tiempo/10e8);
    printf("Norm\u00b2: %.14f\n", norm2);
    
    free(x_new);
    for (int i = 0; i < N; i++) {
        free(a[i]);
    }
    free(a);
    free(b);
    free(x);
    
    return 0;
}
