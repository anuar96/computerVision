#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using namespace std;
const int eps = 0.001;
void grayscale(BMP* image,Matrix<float> & img){
	for (int i = 0; i < image->TellWidth();i++)
		for (int j = 0; j < image->TellHeight();j++){
            RGBApixel pixel = image->GetPixel(i,j);
            img(j,i) = int(0.299 * pixel.Red + 0.587*pixel.Green + 0.114*pixel.Blue);
        }
}
void sobel_hor(const Matrix<float> & img,Matrix<float> & vert){
    vector<float> sobelvert = {1.0,0.0,-1.0};
    int radius = 1;
    float sum = 0.0;
    for (uint i = radius; i < img.n_rows-radius; i++){
        for (uint j = radius; j < img.n_cols - radius; j++){
            for (int k = -radius; k < radius + 1 ; k++){
//                std::cout << i << ' ' << j + k << std::endl;
                sum += sobelvert[k + radius]*img(i,j+k);
            }
        vert(i,j) = sum;
        sum = 0.0;
        }
    }
}
void sobel_vert(const Matrix<float> & img,Matrix<float> & hor){
    vector<float> sobelhor = {-1.0,0.0,1.0};
    int radius = 1;
    float sum = 0.0;
    for (uint i = radius; i < img.n_rows - radius; i++){
        for (uint j = radius; j < img.n_cols-radius; j++){
            for (int k = -radius; k < radius + 1 ; k++){
                sum+=sobelhor[k + radius]*img(i+k,j);
            }
        hor(i,j) = sum;

        sum = 0.0;
        }
    }
}
void normalization(vector<float> & a){
    uint i;
    float norm=0;
//     cout << "size" << a.size() << endl;

   for (i = 0;i < a.size();i++)
           norm+=a[i]*a[i];
   norm = sqrt(norm);
//   if (norm < eps && norm > -eps) 0.969697
    if (norm > eps)
    for (i = 0;i < a.size();i++)
            a[i]=a[i]/norm;
//   cout << ";d;d"<< endl;
 //  for (i=0;i<a.size();i++)
   //   cout << a[i] << endl;
}
