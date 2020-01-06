#ifndef LBP_H
#define LBP_H

#endif // LBP_H
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
const int cellrowslbp = 8;
const int cellcolslbp = 18;

void normalizationlbp(vector<float> & a){
    uint i;
    float norm=0;
//     cout << "size" << a.size() << endl;

   for (i = 0;i < a.size();i++)
           norm+=a[i]*a[i];
   norm = sqrt(norm);
   if (norm > 0.00001)
       for (i = 0;i < a.size();i++)
            a[i]=a[i]/norm;
}

int from_binary_to_decimal(vector<int> binary){
    int decimal=0;
    for (int i = 0; i < 8; i++){
        decimal+=binary[7-i]*pow(2,i);
    }
    return decimal;
}

void LBP(const Matrix<float> & img,vector<float> & descriptor){
    uint i,j,k,z,index;
    uint rows = img.n_rows/cellrowslbp;
    uint cols = img.n_cols/cellcolslbp;
    vector<int> clockwisej = {1,1,0,-1,-1,-1,0,1};
    vector<int> clockwisei = {0,-1,-1,-1,0,1,1,1};

    for (k = 0; k < cellrowslbp;++k){
        for (z = 0; z < cellcolslbp; ++z){
            vector<float> histogram(256,0);
            for (i = 0;i < rows; i++){
                for (j = 0; j < cols; j++){
                    vector<int> binary(8,0);
                    if (((i+k*rows != 0) && (j+z*cols != 0)) && ((i+k*rows<img.n_rows-1) && (j+z*cols < img.n_cols-1))){
                        for (index=0;index < 8; ++index){
                            if (img(i+k*rows,j+z*cols) < img(i+k*rows+clockwisei[index],j+z*cols+clockwisej[index])){
                                binary[index] = 1;
                            }
                        }
                    }
                ++histogram[from_binary_to_decimal(binary)];
                }
            }
            normalizationlbp(histogram);
            for (i = 0;i < 256;i++)
                descriptor.push_back(histogram[i]);
        }
    }
}




