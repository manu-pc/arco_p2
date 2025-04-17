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

    //? Mejora 1: uso de aligned_alloc con el tamaño de línea caché
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

    start_counter();

    for (iter = 0; iter < max_iter; iter++)
    {
        norm2 = 0.0;
        for (int i = 0; i < n; i++)
        {
            float sigma = 0.0;
            //? Mejora 2: se guarda la posición a[i] para no calcularla en todas las iteraciones
            float *a_temp = a[i];

            //? Mejora 3: desdoblamiento del bucle principal sumando los datos de 8 en 8
            //? Mejora 4: separación del bucle en dos etapas, de 'j' a 'i' y de 'i' a 'n'.
            //? Esto permite omitir todas las comparaciones de (i!=j)
            for (int j = 0; j <= i - 8; j += 8)
            {
                sigma += a_temp[j] * x[j];
                sigma += a_temp[j + 1] * x[j + 1];
                sigma += a_temp[j + 2] * x[j + 2];
                sigma += a_temp[j + 3] * x[j + 3];
                sigma += a_temp[j + 4] * x[j + 4];
                sigma += a_temp[j + 5] * x[j + 5];
                sigma += a_temp[j + 6] * x[j + 6];
                sigma += a_temp[j + 7] * x[j + 7];
            }
            // Por si n no es divisible entre 8
            for (int j = i - (i % 8); j < i; j++)
            {
                sigma += a_temp[j] * x[j];
            }

            // Se continúa, de i+1 a n, habiendo saltado el punto de conflicto i==j
            for (int j = i + 1; j <= n - 8; j += 8)
            {
                sigma += a_temp[j] * x[j];
                sigma += a_temp[j + 1] * x[j + 1];
                sigma += a_temp[j + 2] * x[j + 2];
                sigma += a_temp[j + 3] * x[j + 3];
                sigma += a_temp[j + 4] * x[j + 4];
                sigma += a_temp[j + 5] * x[j + 5];
                sigma += a_temp[j + 6] * x[j + 6];
                sigma += a_temp[j + 7] * x[j + 7];
            }
            for (int j = n - (n - i - 1) % 8; j < n; j++)
            {
                if (j > i)
                    sigma += a_temp[j] * x[j];
            }

            x_new[i] = (b[i] - sigma) / a[i][i];
            norm2 += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }

        //? Mejora 5: desdoblamiento del bucle de x=x_new
        for (int i = 0; i < n - 7; i += 8)
        {
            x[i] = x_new[i];
            x[i + 1] = x_new[i + 1];
            x[i + 2] = x_new[i + 2];
            x[i + 3] = x_new[i + 3];
            x[i + 4] = x_new[i + 4];
            x[i + 5] = x_new[i + 5];
            x[i + 6] = x_new[i + 6];
            x[i + 7] = x_new[i + 7];
        }
        for (int i = n - (n % 8); i < n; i++)
        {
            x[i] = x_new[i];
        }

        //? Mejora 6: se omite sqrt porque es una operación más costosa que un producto.
        if (norm2 < tol * tol)
        {
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

    free(a);
    free(b);
    free(x);
    free(x_new);
}
