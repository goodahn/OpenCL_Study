
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library
#include "mnist.hpp"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>

#include <iostream>
#include <fstream>

#include <err_code.h>

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define MATRIX_SIZE 65536*2*2*2*2*2*2*2*2*2

using namespace std;

int main(void)
{
    vector<cl::Platform> platforms;  
    cl::Platform::get(&platforms);  
	vector<cl::Device> devices;  

    int platform_id = 0;
    int device_id = 1;

    std::cout << "Number of Platforms: " << platforms.size() << std::endl;

    for(vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it){
        cl::Platform platform(*it);

        std::cout << "Platform ID: " << platform_id++ << std::endl;  
        std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;  
        std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;  

        platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);  

        for(vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2){
            cl::Device device(*it2);

            std::cout << "\tDevice " << device_id++ << ": " << std::endl;
            std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;  
            std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
            std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;  
            std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
            std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
            std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
            std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
        }  
        std::cout<< std::endl;
    } 
    cout << "Which device do you want to use?" << endl;
    cout << "For using both device type '0'" << endl;

    int selection = 0;
    /*
    while (true) {
        cin >> selection;
        if (selection >=0 && selection <= device_id)
            break;
    }
    */
    cout << "You choose [" << selection << "] device" << endl;

    vector<float> h_A(MATRIX_SIZE);
    vector<float> h_B(MATRIX_SIZE);
    vector<float> h_C(MATRIX_SIZE);
    vector<float> h_C2(MATRIX_SIZE);

    cl::Buffer d_A;
    cl::Buffer d_B;
    cl::Buffer d_C;
    cl::Buffer d_A2;
    cl::Buffer d_B2;
    cl::Buffer d_C2;

    default_random_engine de(time(0));
    normal_distribution<float> nd(1000, 200);

    for (int i=0; i<h_A.size(); i++) {
        h_A[i] = nd(de);
        h_B[i] = nd(de);
        h_C[i] = nd(de);
    }
    
    try {
        if (selection != 0) {
            cl::Context context(devices[selection - 1]);

            cl::Program program(context, util::loadProgram("vadd.cl"), true);

            cl::CommandQueue queue(context);

            d_A = cl::Buffer(context, h_A.begin(), h_A.end(), true);
            d_B = cl::Buffer(context, h_B.begin(), h_B.end(), true);
            d_C = cl::Buffer(context, h_C.begin(), h_C.end(), true);

            cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> vmul(program, "vmul");

            cl::NDRange global_vmul(MATRIX_SIZE);

            queue.finish();

            util::Timer timer;
            double rtime, rtime2;

            vmul(cl::EnqueueArgs(queue, global_vmul),
                    0, MATRIX_SIZE, d_A, d_B, d_C);

            queue.finish();
            rtime = static_cast<double>(timer.getTimeMilliseconds());

            cout << "it takes " << rtime << " milliseconds " << endl;

            cl::copy(queue, d_C, h_C.begin(), h_C.end());
            queue.finish();

            float tmp;
            int correct = 0;
            for (int i=0; i<h_C.size(); i++) {
                tmp = h_A[i]*h_B[i];
                tmp -= h_C[i];
                if (tmp*tmp < TOL*TOL) {
                    correct++;
                }
            }

            cout << "correct " << correct << " / " << MATRIX_SIZE << endl;
        } else {
            cl::Context context(devices);

            cl::Program program(context, util::loadProgram("vadd.cl"), true);

            cl::CommandQueue queue(context, devices[0]);
            cl::CommandQueue queue2(context, devices[1]);

            d_A = cl::Buffer(context, h_A.begin(), h_A.end() - h_A.size()/4, true);
            d_B = cl::Buffer(context, h_B.begin(), h_B.end() - h_B.size()/4, true);
            d_C = cl::Buffer(context, h_C.begin(), h_C.end() - h_C.size()/4, true);
            d_A2 = cl::Buffer(context, h_A.end() - h_A.size()/4, h_A.end(), true);
            d_B2 = cl::Buffer(context, h_B.end() - h_B.size()/4, h_B.end(),  true);
            d_C2 = cl::Buffer(context, h_C.begin(), h_C.end() - h_C.size()/4*3, true);

            cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> vmul(program, "vmul");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer, cl::Buffer> vmul2(program, "vmul");

            cl::NDRange global_vmul(MATRIX_SIZE);
            queue.finish();
            queue2.finish();

            util::Timer timer;
            double rtime, rtime2;

            vmul(cl::EnqueueArgs(queue, global_vmul),
                    0, MATRIX_SIZE/4*3, d_A, d_B, d_C);
            vmul2(cl::EnqueueArgs(queue2, global_vmul),
                    0, MATRIX_SIZE/4, d_A2, d_B2, d_C2);

            queue.finish();
            //queue2.finish();
            rtime = static_cast<double>(timer.getTimeMilliseconds());

            cout << "[result2]it takes " << rtime << " milliseconds " << endl;

            cl::copy(queue, d_C, h_C.begin(), h_C.end()-h_C.size()/4);
            cl::copy(queue2, d_C2, h_C2.begin(), h_C2.end()-h_C2.size()/4*3);
            queue.finish();

            float tmp;
            int correct = 0;
            for (int i=0; i<h_C.size()/4*3; i++) {
                tmp = h_A[i]*h_B[i];
                tmp -= h_C[i];
                if (tmp*tmp < TOL*TOL) {
                    correct++;
                }
            }
            for (int i=0; i<h_C2.size()/4; i++) {
                tmp = h_A[i+h_C.size()/4*3]*h_B[i+h_C.size()/4*3];
                tmp -= h_C2[i];
                if (tmp*tmp < TOL*TOL) {
                    correct++;
                }
            }

            cout << "correct " << correct << " / " << MATRIX_SIZE << endl;
        }
    } catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr 
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
           << ")"
           << std::endl;
    }

    return 0;
}
