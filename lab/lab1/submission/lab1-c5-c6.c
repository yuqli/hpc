//
// Created by YUQIONG LI on 8/10/2018.
//
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <mkl.h>

int main(){
    // hyperparameters
    int INPUT_1 = 25;
    int INPUT_2 = 25;
    int HIDDEN = 400;
    int OUT = 100;

    /*
     * C5 : naive c
     */

    struct timespec start, end;

    // 1. initialization
    double x[INPUT_1][INPUT_2];
    int i;  // row indexer
    int j;  // column indexer
    for (i = 0; i < INPUT_1; i++){
        for (j = 0; j < INPUT_2; j++)
            x[i][j] = 0.4 + ((i + j) % 40 - 20) / 40.0;
    }

    double weights1[HIDDEN][INPUT_1 * INPUT_2];
    for (i = 0; i < HIDDEN; i++){
        for (j = 0; j < INPUT_1 * INPUT_2; j++)
            weights1[i][j] = 0.4 + ((i + j) % 40 - 20) / 40.0;
    }

    double weights2[OUT][HIDDEN];
    for (i = 0; i < OUT; i++){
        for (j = 0; j < HIDDEN; j++)
            weights2[i][j] = 0.4 + ((i + j) % 40 - 20) / 40.0;
    }

    // 2. inferencing
    clock_gettime(CLOCK_MONOTONIC, &start);

    double x_flat[INPUT_1 * INPUT_2];
    for (i = 0; i < INPUT_1; i++){
        for (j = 0; j < INPUT_2; j++)
            x_flat[i * INPUT_1 + j] = x[i][j];
    }

    double hidden[HIDDEN];
    for (i = 0; i < HIDDEN; i++){
        double res = 0.0;  // to store the res of weighted sum
        for (j = 0; j < INPUT_1 * INPUT_2; j++)
            res += weights1[i][j] * x_flat[j];
        hidden[i] = res;
    }

    double out[OUT];
    for (i = 0; i < OUT; i++){
        double res = 0.0;
        for (j = 0; j < HIDDEN; j++)
            res += weights2[i][j] * hidden[j];
        out[i] = res;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_usec = ((double)end.tv_sec - (double)start.tv_sec) * pow(10, 3) +
                       ((double)end.tv_nsec - (double)start.tv_nsec) * pow(10, -6);
    // time_usec *= pow(10, -6);

    // 3. final output
    double res;
    for (i = 0; i < OUT; i++)
        res += out[i];

    printf("Naive C result: %.2f\n", res);
    printf("Running time: %.2f millseconds\n", time_usec);


    /*
     * C6: math library C
     */


    return 0;
}


