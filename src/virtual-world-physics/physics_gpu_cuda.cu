#include "physics_gpu.h"
#include "physics_gpu.cuh"

// Define here kernels and entry points for GPU physics computations.

__global__ void generateMortonCodes(unsigned int *codes, glm::vec3 *centers, int *ids, const int& n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n)
        return;

    codes[tid] = morton3D(centers[tid].x, centers[tid].y, centers[tid].z);
    ids[tid] = tid;
}

namespace physics {

bool GPUCollisionDetector::Initialize() {
    return false;
}

std::vector<CollisionInfo> GPUCollisionDetector::DetectCollisions(const std::vector<PhysicsObject>& objects) {
    const int n = objects.size();
    const int blockSize = 128;
    const int gridSize = (n + blockSize - 1) / blockSize;

    std::vector<glm::vec3> aabbs;
    aabbs.reserve(n);

    for (auto& obj : objects) {
        aabbs.push_back(obj.boundingVolume.center);
    }
    
    cudaError_t err;

    unsigned int *inputMortonCodes;
    err = cudaMalloc(&inputMortonCodes, n * sizeof(unsigned int));
    checkCudaError(err);

    glm::vec3 *aabbCenters;
    err = cudaMalloc(&aabbCenters, n * sizeof(glm::vec3));
    checkCudaError(err);
    err = cudaMemcpy(aabbCenters, aabbs.data(), n * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCudaError(err);

    int *initialObjectIDs;
    err = cudaMalloc(&initialObjectIDs, n * sizeof(int));
    checkCudaError(err);

    // Acquire morton codes
    generateMortonCodes<<<gridSize, blockSize>>>(inputMortonCodes, aabbCenters, initialObjectIDs, n);
    checkCudaError(cudaGetLastError());
    cudaDeviceSynchronize();

    // Sort object indices using theirs morton codes as sorting keys (NOTE: maybe implement your own radix sort? check leetgpu)
    size_t tempStorage = 0;
    void *d_tempStorage = nullptr;

    unsigned int *sortedMortonCodes;
    err = cudaMalloc(&sortedMortonCodes, n * sizeof(unsigned int));
    checkCudaError(err);

    int *sortedObjectIDs;
    err = cudaMalloc(&sortedObjectIDs, n * sizeof(int));
    checkCudaError(err);

    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorage, inputMortonCodes, sortedMortonCodes, initialObjectIDs, sortedObjectIDs, n);
    checkCudaError(cudaGetLastError());
    
    err = cudaMalloc(&d_tempStorage, tempStorage);
    checkCudaError(err);

    cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorage, inputMortonCodes, sortedMortonCodes, initialObjectIDs, sortedObjectIDs, n);
    checkCudaError(cudaGetLastError());

    // TODO: actually build + query


    // Free everything
    err = cudaFree(inputMortonCodes);
    checkCudaError(err);

    err = cudaFree(aabbCenters);
    checkCudaError(err);

    err = cudaFree(initialObjectIDs);
    checkCudaError(err);

    err = cudaFree(d_tempStorage);
    checkCudaError(err);

    err = cudaFree(sortedMortonCodes);
    checkCudaError(err);

    err = cudaFree(sortedObjectIDs);
    checkCudaError(err);

    return {};
}

}