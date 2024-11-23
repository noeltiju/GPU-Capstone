#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tiffio.h>
#include <vector>
#include <algorithm>
#include <cstdint> 
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;

__host__ cudnnHandle_t createCudaHandleAndOutputHWSpecs()
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); 
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;
    return handle_;
}

__host__ std::tuple<cudnnTensorDescriptor_t, float*, int, int, int> loadImageAndPreprocess(const char* filePath)
{
    TIFF* tiff = TIFFOpen(filePath, "r");
    if (!tiff) {
        cerr << "Error: Could not open TIFF file" << endl;
        exit(1);
    }

    uint32_t width, height;
    size_t npixels;
    uint32_t* raster;

    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    npixels = width * height;

    raster = (uint32_t*) _TIFFmalloc(npixels * sizeof(uint32_t));
    if (raster == NULL) {
        cerr << "Error: Could not allocate memory for raster" << endl;
        exit(1);
    }

    if (!TIFFReadRGBAImage(tiff, width, height, raster, 0)) {
        cerr << "Error: Could not read TIFF image" << endl;
        exit(1);
    }
    TIFFClose(tiff);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, height, width);

    float* input_data;
    cudaMallocManaged(&input_data, npixels * 3 * sizeof(float)); 

    for (size_t i = 0; i < npixels; ++i) {
        input_data[i * 3 + 0] = (float) TIFFGetR(raster[i]) / 255.0f; 
        input_data[i * 3 + 1] = (float) TIFFGetG(raster[i]) / 255.0f; 
        input_data[i * 3 + 2] = (float) TIFFGetB(raster[i]) / 255.0f; 
    }

    _TIFFfree(raster);
    return {input_desc, input_data, 1, (int)height, (int)width};
}

__host__ float* runCuDnnModel(cudnnHandle_t handle_, cudnnTensorDescriptor_t input_desc, float* input_data, int num_classes)
{
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_classes, 1, 1);

    float* output_data;
    cudaMallocManaged(&output_data, num_classes * sizeof(float));

    for (int i = 0; i < num_classes; ++i) {
        output_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    return output_data;
}

__host__ void printClassificationResults(ofstream& output_file, const string& file_name, float* output_data, int num_classes)
{
    int max_idx = std::max_element(output_data, output_data + num_classes) - output_data;
    output_file << "File: " << file_name << "\n";
    output_file << "Predicted class: " << max_idx << "\n";
    output_file << "Class probabilities: ";
    for (int i = 0; i < num_classes; ++i) {
        output_file << output_data[i] << " ";
    }
    output_file << "\n\n";
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_folder> <output_file.txt>" << endl;
        return 1;
    }

    const string input_folder = argv[1];
    const string output_file_path = argv[2];
    int num_classes = 10; 

    cudnnHandle_t handle_ = createCudaHandleAndOutputHWSpecs();
    ofstream output_file(output_file_path);

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".tiff" || entry.path().extension() == ".tif") {
            auto [input_desc, input_data, n, h, w] = loadImageAndPreprocess(entry.path().c_str());
            float* output_data = runCuDnnModel(handle_, input_desc, input_data, num_classes);
            printClassificationResults(output_file, entry.path().string(), output_data, num_classes);

            cudaFree(input_data);
            cudaFree(output_data);
        }
    }

    cudnnDestroy(handle_);
    std::cout << "Destroyed cuDNN handle." << std::endl;
    output_file.close();

    return 0;
}
