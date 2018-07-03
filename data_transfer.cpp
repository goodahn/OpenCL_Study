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

using namespace std;

int main(void)
{
    vector<cl::Platform> platforms;  
    cl::Platform::get(&platforms);  
	vector<cl::Device> devices;  

    int platform_id = 0;
    int device_id = 0;

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
    int selection = -1;
    while (true) {
        cin >> selection;
        if (selection >=0 && selection <= device_id)
            break;
    }
    cout << "You choose [" << selection << "] device" << endl;

    try 
    {
    	// Create a context
        cl::Context context(devices[selection]);
        cout << "Create context succed" << endl;

        cl::CommandQueue queue(context);
        cl::Buffer d_X;
        queue.finish();
        cout << "Data transferring test starts!" << endl;

        for (int i=1;i<900;i++) {
            util::Timer timer;
            vector<char> tmp(768*4);
            vector<char> tmp2(768*4*60);
            vector<char> tmp3(10*60*4);
            d_X = cl::Buffer(context, tmp.begin(), tmp.end(), true);
            queue.finish();
            double rtime = static_cast<double>(timer.getTimeMilliseconds());
            cout << "Transferring " << i*100 << " bytes ends in " << rtime << " milli seconds!" <<endl;
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
}
