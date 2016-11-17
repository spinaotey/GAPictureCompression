#include "gnal.h"
#include <inttypes.h>
#include <linux/random.h>
#include <stdio.h>
#include <stdlib.h>
#include <syscall.h>
#include <unistd.h>


/*  RUNIF
 *  
 *  Using /dev/urandom, generates a random number
 *  between 0 and 1, without including 1.
 *
 *  -Return: double from 0 up to 1.
 */
double rUnif(void){
    uint64_t a;             //random number from 0 to 2^63-1
    uint64_t mask = 1;
    uint64_t max = 1;
    mask = ~(mask << 63);   //Creates a mask with 011...1
    max = max <<63;         //Maximum number, 2^63
    syscall(SYS_getrandom,&a,sizeof(uint64_t),GRND_NONBLOCK);
    a = a & mask;
    return (double)a/(double)max;
}

/*poly_t newPoly(double xmax, double ymax){
    
}*/
