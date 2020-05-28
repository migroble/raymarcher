#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include <CL/cl.h>
#include <CL/cl_platform.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ocl.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
double get_time() {
    static struct timeval   tv0;
    double time_, time;

    gettimeofday(&tv0,(struct timezone*)0);
    time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
    time = time_/1000000;
    return(time);
}
#pragma GCC diagnostic pop

void raymarch(unsigned char *image, int width, int height) {
    cl_int err;
    
    cl_device_id     device_id;
    cl_context       context;
    cl_command_queue commands;
    cl_kernel        k_render;
    
    cl_mem d_image;
    cl_mem d_spheres;
    
    size_t global[2];
    
    ocl_setup(DEVICE, &device_id, &context, &commands);
    check_error(ocl_output_device_info(device_id));
    
    k_render = ocl_compile_kernel("raymarcher.cl", "render", device_id, context);
    
    d_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 3 * sizeof(unsigned char) * width * height, NULL, &err);
    check_error(err);
    
    ocl_set_kernel_args(k_render, (arg_t[]){ arg(d_image), arg(width), arg(height) }, 3);
    
    global[0] = width;
    global[1] = height;
    check_error(clEnqueueNDRangeKernel(commands, k_render, 2, NULL, global, NULL, 0, NULL, NULL));
    check_error(clFinish(commands));
    
    check_error(clEnqueueReadBuffer(commands, d_image, CL_TRUE, 0, 3 * sizeof(unsigned char) * width * height, image, 0, NULL, NULL));
    
    clReleaseMemObject(d_image);
    clReleaseKernel(k_render);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
}

int main(int argc, char **argv) {
    int width = 0, height = 0;
    double t0, t1;
    
    if (argc == 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    
    if (width == 0)
        width = 3840;
    if (height == 0)
        height = 2160;
    
    unsigned char *image = (unsigned char *)malloc(3 * sizeof(unsigned char) * width * height);
    
    t0 = get_time();
    raymarch(image, width, height);
    t1 = get_time();
    
    printf("OCL Exection time %f ms.\n", t1 - t0);
    
    stbi_write_png("out.png", width, height, 3, image, 3 * width);
    
    free(image);
}
