// Main convolutional algoirthm

#include <math.h>
#include <time.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "imageLoader.cpp"

#define GRIDSIZE 16

void convolution_cpu(const byte *orig, byte *cpu, const unsigned int cols,
                     const unsigned int rows);

__global__ void convolution_gpu(const byte *orig, byte *gpu,
                                const unsigned int cols,
                                const unsigned int rows) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  float dx, dy;
  if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1) {
    dx = (-1 * orig[(y - 1) * cols + (x - 1)]) +
         (-2 * orig[y * cols + (x - 1)]) +
         (-1 * orig[(y + 1) * cols + (x - 1)]) +
         (orig[(y - 1) * cols + (x + 1)]) + (2 * orig[y * cols + (x + 1)]) +
         (orig[(y + 1) * cols + (x + 1)]);
    dy = (orig[(y - 1) * cols + (x - 1)]) + (2 * orig[(y - 1) * cols + x]) +
         (orig[(y - 1) * cols + (x + 1)]) +
         (-1 * orig[(y + 1) * cols + (x - 1)]) +
         (-2 * orig[(y + 1) * cols + x]) +
         (-1 * orig[(y + 1) * cols + (x + 1)]);
    gpu[y * cols + x] = sqrt((dx * dx) + (dy * dy));
  }
}

int main(int argc, char *argv[]) {
  imageData img_original = loadImage(argv[1]);
  imageData img_cpu(new byte[img_original.cols * img_original.rows],
                    img_original.cols, img_original.rows);
  imageData img_gpu(new byte[img_original.cols * img_original.rows],
                    img_original.cols, img_original.rows);

  memset(img_cpu.pixels, 0, (img_original.cols * img_original.rows));

  auto c = std::chrono::system_clock::now();
  convolution_cpu(img_original.pixels, img_cpu.pixels, img_original.cols,
                  img_original.rows);
  std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - c;

  byte *gpu_orig, *gpu_convolution;
  cudaMalloc((void **)&gpu_orig, (img_original.cols * img_original.rows));
  cudaMalloc((void **)&gpu_convolution,
             (img_original.cols * img_original.rows));
  cudaMemcpy(gpu_orig, img_original.pixels,
             (img_original.cols * img_original.rows), cudaMemcpyHostToDevice);
  cudaMemset(gpu_convolution, 0, (img_original.cols * img_original.rows));

  dim3 threadsPerBlock(GRIDSIZE, GRIDSIZE, 1);
  dim3 numBlocks(ceil(img_original.cols / GRIDSIZE),
                 ceil(img_original.rows / GRIDSIZE), 1);

  c = std::chrono::system_clock::now();
  convolution_gpu<<<numBlocks, threadsPerBlock>>>(
      gpu_orig, gpu_convolution, img_original.cols, img_original.rows);
  cudaError_t cudaerror = cudaDeviceSynchronize();
  fprintf(stderr, "Cuda failed to synchronize: %s\n",
          cudaGetErrorName(cudaerror));
  std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;
  cudaMemcpy(img_gpu.pixels, gpu_convolution,
             (img_original.cols * img_original.rows), cudaMemcpyDeviceToHost);

  writeImage(argv[1], "gpu", img_gpu);
  writeImage(argv[1], "cpu", img_cpu);

  cudaFree(gpu_orig);
  cudaFree(gpu_convolution);
  return 0;
}

void convolution_cpu(const byte *orig, byte *cpu, const unsigned int cols,
                     const unsigned int rows) {
  for (int y = 1; y < rows - 1; y++) {
    for (int x = 1; x < cols - 1; x++) {
      int dx = (-1 * orig[(y - 1) * cols + (x - 1)]) +
               (-2 * orig[y * cols + (x - 1)]) +
               (-1 * orig[(y + 1) * cols + (x - 1)]) +
               (orig[(y - 1) * cols + (x + 1)]) +
               (2 * orig[y * cols + (x + 1)]) +
               (orig[(y + 1) * cols + (x + 1)]);
      int dy = (orig[(y - 1) * cols + (x - 1)]) +
               (2 * orig[(y - 1) * cols + x]) +
               (orig[(y - 1) * cols + (x + 1)]) +
               (-1 * orig[(y + 1) * cols + (x - 1)]) +
               (-2 * orig[(y + 1) * cols + x]) +
               (-1 * orig[(y + 1) * cols + (x + 1)]);
      cpu[y * cols + x] = sqrt((dx * dx) + (dy * dy));
    }
  }
}
