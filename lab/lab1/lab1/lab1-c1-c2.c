//
// Created by YUQIONG LI on 4/10/2018.
//
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
    int ARRAY_SIZE = (int) (3 * pow(2, 28));
    int *array = malloc(ARRAY_SIZE * sizeof(int));
    int i;
    for (i = 0; i < ARRAY_SIZE; ++i)
        array[i] = (double) i / 3;

    double times[10];  // store runtimes
    int run; // index for which run
    double sum = 0;
    struct timespec start, end;

    /* Experiment 1 */
    for (run = 0; run < 10; run++){
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (i = 0; i < ARRAY_SIZE; ++i)
            sum+= array[i] * 2;

        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_usec = ((double)end.tv_sec - (double)start.tv_sec) * pow(10, 6) +
		           ((double)end.tv_nsec - (double)start.tv_nsec) * pow(10, -3);
        times[run] = time_usec;
    }

    // get results
    double best = times[0];
    for (i = 1; i < 10; i++)
        best = times[i] < best? times[i] : best;

    best = best * pow(10, -6);
    double bw = 3 / best;
    double flops = ARRAY_SIZE * 2 * pow(10, -9) / best;
    printf("*C1*\n");
    printf("Time: %.2f secs\n", best);
    printf("BW: %.2f GB/s\n", bw);
    printf("FLOPS: %2f GFLOP/s\n", flops);


    /*  Experiment 2 */

    for (run = 0; run < 10; run++){
        sum = 0;
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (i = 0; i < ARRAY_SIZE; i+=8){
            sum += array[i] * 2;
            sum += array[i+1] * 2;
            sum += array[i+2] * 2;
            sum += array[i+3] * 2;
            sum += array[i+4] * 2;
            sum += array[i+5] * 2;
            sum += array[i+6] * 2;
            sum += array[i+7] * 2;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_usec = ((double)end.tv_sec - (double)start.tv_sec) * 1000000 +
                           ((double)end.tv_nsec - (double)start.tv_nsec) / 1000;
        times[run] = time_usec;
    }

    best = times[0];
    for (i = 1; i < 10; i++)
        best = times[i] < best? times[i] : best;

    best = best * pow(10, -6);
    bw = 3 / best;
    flops = ARRAY_SIZE * 2 * pow(10, -9) / best;
    printf("*C2*\n");
    printf("Time: %.2f secs\n", best);
    printf("BW: %.2f GB/s\n", bw);
    printf("FLOPS: %2f GFLOP/s\n", flops);

    return 0;
}
