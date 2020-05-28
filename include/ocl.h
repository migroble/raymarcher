#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

typedef struct arg_t {
    size_t size;
    void *val;
} arg_t;
#define arg(v) (arg_t){ sizeof((v)), (void *)&(v) }

char *err_code(cl_int err_in);
#define check_error(err) { chck_err(err, __func__, __FILE__, __LINE__); }
inline void chck_err(cl_int err, const char *func, const char *file, int line) {
    if (err != CL_SUCCESS) {
        printf("%s(%s:%d): %s\n", func, file, line, err_code(err));
        
        exit(-1);
    }
}

cl_int ocl_output_device_info(cl_device_id device_id);

void ocl_setup(cl_device_type device_type, cl_device_id *device_id, cl_context *context, cl_command_queue *commands);

cl_kernel ocl_compile_kernel(const char *filename, const char *kernel_name, cl_device_id device_id, cl_context context);

void ocl_set_kernel_args(cl_kernel kernel, arg_t *args, int n_args);
