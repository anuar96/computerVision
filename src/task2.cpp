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
#include "base.h"
#include "lbp.h"
#include "colorfeatures.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::get;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
const int cellrows = 8;
const int cellcols = 8;
const int segments = 8;
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }
    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
           // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}
// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    uint i,j;
    uint k,z;

    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        Matrix<float> img(data_set[image_idx].first->TellHeight(),data_set[image_idx].first->TellWidth());
        vector<float> featuresofcolor = colorfeatures(data_set[image_idx].first);
        grayscale(data_set[image_idx].first,img);

        vector <float> lbp;
        LBP(img,lbp);

        Matrix<float> vertical(img.n_rows,img.n_cols);
        sobel_vert(img,vertical);
        Matrix<float> horizontal(img.n_rows,img.n_cols);
        sobel_hor(img,horizontal);
        Matrix<float> map(img.n_rows,img.n_cols);
        for (i = 1; i < img.n_rows-1; i++){
            for (j = 1; j < img.n_cols-1; j++){
                map(i,j) = sqrt(vertical(i,j)*vertical(i,j)+horizontal(i,j)*horizontal(i,j));
            }
        }
        for (i = 1; i < img.n_rows-1; i++){
            for (j = 1; j < img.n_cols-1;j++){
                img(i,j) = atan2(vertical(i,j),horizontal(i,j));
            }
        }

        img = img.submatrix(1,1,img.n_rows-2,img.n_cols-2);
        map = map.submatrix(1,1,map.n_rows-2,map.n_cols-2);
        uint rows = img.n_rows/cellrows;
        uint cols = img.n_cols/cellcols;
        float dlina = 2*M_PI/segments;
        vector<float> a;
        for (k = 0;k < cellrows; k++){
            for (z = 0; z < cellcols; z++){
                vector<float> b(segments,0);
                for (i = 0; i < rows;i++){
                    for ( j = 0; j < cols;j++){
                        b[int((img(i+k*rows,j+z*cols)+M_PI)/dlina)%segments] += map(i+k*rows,j+z*cols);
                    }
                }
                normalization(b);
                for (i=0;i<segments;++i)
                    a.push_back(b[i]);
            }
        }
       
        for (i=0;i<lbp.size();i++){
            a.push_back(lbp[i]);
        }
        for (i=0;i<featuresofcolor.size();i++)
            a.push_back(featuresofcolor[i]);
        features->push_back(make_pair(a,data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier

    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);
        // PLACE YOUR CODE HERE
   // You can change parameters of classifier here
//
   // for (int i = 0; i < data_set.size();++i){
 //       grayscale(&data_set[i].first);
//    }
    params.C = 0.01;
    TClassifier classifier(params);
//	cout << "um here!2!#!@!#" << endl;
    // Train classifier
    classifier.Train(features, &model);

    // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser


	ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");
        // Parse options


    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train){
    	TrainClassifier(data_file, model_file);
    }
    	// If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
