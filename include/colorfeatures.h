#ifndef COLORFEATURES_H
#define COLORFEATURES_H

#endif // COLORFEATURES_H
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
#include <tuple>
const int cellrowscolorfeatures = 8;
const int cellcolscolorfeatures = 8;

Matrix<std::tuple<int,int,int>> transposed(BMP* img){
    Matrix <std::tuple<int,int,int>> transposed(img->TellHeight(),img->TellWidth());
    for (uint i = 0; i < transposed.n_cols; i++){
        for (uint j = 0; j < transposed.n_rows; j++){
            RGBApixel pixel = img->GetPixel(i,j);
            transposed(j,i) = make_tuple(pixel.Red,pixel.Green,pixel.Blue);
        }
    }
    return transposed;
}

vector<float> colorfeatures(BMP* img){
    uint k,z,i,j;
    uint rows = img->TellHeight()/cellrowscolorfeatures;
    uint cols = img->TellWidth()/cellcolscolorfeatures;
    Matrix<std::tuple<int,int,int>> image = transposed(img);
    float norm = 255*rows*cols;
    vector<float> colorfeatures;
    for (k = 0; k < cellrowscolorfeatures; k++){
        for (z = 0;z < cellcolscolorfeatures; z++){
            float sred=0,sblue=0,sgreen=0;
            for (i = 0;i < rows;i++){
                for (j = 0;j < cols;j++){
                    sred+=get<0>(image(i,j));
                    sgreen+=get<1>(image(i,j));
                    sblue+=get<2>(image(i,j));
                }
            }
            sred = sred/norm;
            sgreen = sgreen/norm;
            sblue = sblue/norm;
            colorfeatures.push_back(sred);
            colorfeatures.push_back(sgreen);
            colorfeatures.push_back(sblue);
        }
    }
    return colorfeatures;
}

