#include <inttypes.h>


/* POLYGON */
typedef struct{
    uint8_t nVert;      //Number of vertices
    uint32_t *x, *y;    //Coordinates of vertices
    uint8_t rgba[4];    //RGBA to fill polygon
}poly_t;

/*  RUNIF
 *  
 *  Using /dev/urandom, generates a random number
 *  between 0 and 1, without including 1.
 *
 *  -Return: double from 0 up to 1.
 */
double rUnif(void);
