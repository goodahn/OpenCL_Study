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
#define EPOCH (1000)
#define NUM_LABELS (10)
#define STEP_SIZE (0.5)
#define BATCH_SIZE (60)

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

    int selection = -1;
    while (true) {
        cin >> selection;
        if (selection >=0 && selection <= device_id)
            break;
    }
    cout << "You choose [" << selection << "] device" << endl;

    MnistData data = MnistData();
    data.read_label_and_image("./train-labels-idx1-ubyte",
                              "./train-images-idx3-ubyte",
                              "./t10k-labels-idx1-ubyte",
                              "./t10k-images-idx3-ubyte");

    int image_size = data.get_image_size();

    std::vector<float> h_W(image_size*NUM_LABELS);
    std::vector<float> h_B(NUM_LABELS*BATCH_SIZE);
    std::vector<float> h_X(image_size*BATCH_SIZE);
    std::vector<float> h_Y(NUM_LABELS*BATCH_SIZE);
    std::vector<float> h_Y_(NUM_LABELS*BATCH_SIZE);

    cl::Buffer d_W;
    cl::Buffer d_W_;
    cl::Buffer d_B;
    cl::Buffer d_X;
    cl::Buffer d_Y;
    cl::Buffer d_Y_;
    cl::Buffer d_SUM;
    cl::Buffer d_RESULT;

    default_random_engine de(time(0));
    normal_distribution<float> nd(0, 0.1);

    for (int i=0; i<h_B.size(); i++) {
        h_B[i] = 0.1;
    }

    for (int i=0; i<h_W.size(); i++) {
        h_W[i] = nd(de);
    }

    try 
    {
    	// Create a context
        if (selection != 0) {
            cl::Context context(devices[selection - 1]);
            cout << "Create context succed" << endl;

            // Load in kernel source, creating a program object for the context

            cl::Program program(context, util::loadProgram("matrix.cl"), true);
            cout << "Create Program succed" << endl;

            // Get the command queue
            cl::CommandQueue queue(context);

            // Create the kernel functor

            cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mmul(program, "mmul");
            cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mmul2(program, "mmul2");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer> madd(program, "madd");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mexp(program, "mexp");
            cl::make_kernel<int, int, float, cl::Buffer, cl::Buffer> msub(program, "msub");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mnormalize(program, "mnormalize");

            d_W = cl::Buffer(context, h_W.begin(), h_W.end(), true);
            d_W_ = cl::Buffer(context, h_W.begin(), h_W.end(), true);
            d_B = cl::Buffer(context, h_B.begin(), h_B.end(), true);
            d_Y = cl::Buffer(context, h_Y.begin(), h_Y.end(), true);
            d_SUM = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float));
            d_RESULT = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float));

            queue.finish();
            cout << "Buffer creation finished" << endl;

            util::Timer timer;
            double rtime, rtime2;

            /*
             * Train part
             */

            cl::NDRange global_mmul(image_size);
            cl::NDRange global_mexp(NUM_LABELS*BATCH_SIZE);
            cl::NDRange global_mmul2(BATCH_SIZE);
            cl::NDRange global_msub(NUM_LABELS*image_size);

            for (int i=0;i<EPOCH;i++) {
                rtime2 = static_cast<double>(timer.getTimeMilliseconds()) ;
                vector<char> tmp_images = data.get_batch_train_images(BATCH_SIZE);
                vector<float> tmp_labels = data.get_batch_train_labels(BATCH_SIZE/10);

                d_X = cl::Buffer(context, tmp_images.begin(), tmp_images.end(), true);
                d_Y_ = cl::Buffer(context, tmp_labels.begin(), tmp_labels.end(), true);

#ifdef ITER_DEBUG
                queue.finish();
                rtime = static_cast<double>(timer.getTimeMilliseconds())  - rtime2;
                rtime2 = static_cast<double>(timer.getTimeMilliseconds()) ;
                cout << i << " Iteration buffer created it takes " << rtime << " milliseconds" << endl;
#endif
                /*
                 * Y = Wx + b
                 */
                mmul(cl::EnqueueArgs(queue, global_mmul),
                        0, image_size, NUM_LABELS, BATCH_SIZE, d_W, d_X, d_Y);
                madd(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_B, d_Y);
#ifdef ITER_DEBUG
                queue.finish();
                rtime = static_cast<double>(timer.getTimeMilliseconds())  - rtime2;
                rtime2 = static_cast<double>(timer.getTimeMilliseconds()) ;
                cout << i << " Iteration finish Wx+bit takes " << rtime << " milliseconds" << endl;
#endif

                /*
                 * softmax
                 */
                mexp(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_Y, d_SUM);
                mnormalize(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_Y, d_SUM);
#ifdef ITER_DEBUG
                queue.finish();
                rtime = static_cast<double>(timer.getTimeMilliseconds())  - rtime2;
                rtime2 = static_cast<double>(timer.getTimeMilliseconds()) ;
                cout << i << " Iteration finish softmaxit takes " << rtime << " milliseconds" << endl;
#endif

                /*
                 * update weight with gradient descent algorithm
                 *  - its cost function is cross entropy
                 */
                msub(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, 1.0, d_Y_, d_Y);
#ifdef ITER_DEBUG
                queue.finish();
                rtime = static_cast<double>(timer.getTimeMilliseconds())  - rtime2;
                rtime2 = static_cast<double>(timer.getTimeMilliseconds()) ;
                cout << i << " Iteration mexp finishedit takes " << rtime << " milliseconds" << endl;
#endif
                mmul2(cl::EnqueueArgs(queue, global_mmul2),
                        0, BATCH_SIZE, NUM_LABELS, image_size, d_Y, d_X, d_W_);
#ifdef ITER_DEBUG
                queue.finish();
                rtime = static_cast<double>(timer.getTimeMilliseconds())  - rtime2;
                rtime2 = static_cast<double>(timer.getTimeMilliseconds()) ;
                cout << i << " Iteration mmul finishedit takes " << rtime << " milliseconds" << endl;
#endif
                msub(cl::EnqueueArgs(queue, global_msub),
                        0, NUM_LABELS*image_size, 0.5, d_W_, d_W);
#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration finished" << endl;
#endif
            }

            queue.finish();

            rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            printf("\nTrain ends in %lf seconds\n", rtime);

            int correct_num = 0;
            /*
             * Check accuracy of model
             */
            for (int i=0; i<data.get_test_size()/BATCH_SIZE; i++) {
                vector<char> tmp_images = data.get_batch_test_images(BATCH_SIZE);
                vector<float> tmp_labels = data.get_batch_test_labels(BATCH_SIZE);
                d_X = cl::Buffer(context, tmp_images.begin(), tmp_images.end(), true);
                d_Y_ = cl::Buffer(context, tmp_labels.begin(), tmp_labels.end(), true);
                /*
                 * Y = Wx + b
                 */
                mmul(cl::EnqueueArgs(queue, global_mmul),
                        0, image_size, NUM_LABELS, BATCH_SIZE, d_W, d_X, d_Y);
                madd(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_B, d_Y);

                /*
                 * softmax
                 */
                mexp(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_Y, d_SUM);
                mnormalize(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_Y, d_SUM);

                /*
                 * check whether calculated answer is correct
                 */
                cl::copy(queue, d_Y, h_Y.begin(), h_Y.end());
                for (int j=0; j<BATCH_SIZE; j++) {
                    int idx = j*NUM_LABELS;
                    float max_p = 0.0f;
                    int max_idx = -1;
                    int sol_idx = -1;
                    for (int k=0; k<NUM_LABELS; k++) {
                        if (max_p < h_Y[idx+k]) {
                            max_p = h_Y[idx+k];
                            max_idx = idx+k;
                        } 
                        if (tmp_labels[idx+k] == 1) {
                            sol_idx = idx+k;
                        }
                    }
                    if (sol_idx == max_idx) {
                        correct_num += 1;
                    }
                }
            }

            queue.finish();

            printf("Accuracy : %d / %d\n", correct_num, data.get_test_size());

            rtime2 = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            printf("\nTest ends in %lf seconds\n", rtime2);


            data.CleanUp();
        } else {
            /*
            cl::Context context(devices[0]);
            cl::Context context2(devices[1]);
            */
            cl::Context context(devices);
            cout << "Create contexts succed" << endl;

            // Load in kernel source, creating a program object for the context

            cl::Program program(context, util::loadProgram("matrix.cl"), true);
            // cl::Program program2(context2, util::loadProgram("matrix.cl"), true);
            cout << "Create Programs succed" << endl;

            // Get the command queue
            cl::CommandQueue queue(context, devices[0]);
            cl::CommandQueue queue2(context, devices[1]);

            // Create the kernel functor

            cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mmul(program, "mmul");
            cl::make_kernel<int, int, int, int, cl::Buffer, cl::Buffer, cl::Buffer> mmul2(program, "mmul2");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer> madd(program, "madd");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mexp(program, "mexp");
            cl::make_kernel<int, int, float, cl::Buffer, cl::Buffer> msub(program, "msub");
            cl::make_kernel<int, int, cl::Buffer, cl::Buffer> mnormalize(program, "mnormalize");

            d_W = cl::Buffer(context, h_W.begin(), h_W.end(), true);
            d_W_ = cl::Buffer(context, h_W.begin(), h_W.end(), true);
            d_B = cl::Buffer(context, h_B.begin(), h_B.end(), true);
            d_Y = cl::Buffer(context, h_Y.begin(), h_Y.end(), true);
            d_SUM = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float));
            d_RESULT = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float));

            queue.finish();
            cout << "Buffer creation finished" << endl;

            util::Timer timer;

            /*
             * Train part
             */

            cl::NDRange global_mmul(image_size);
            cl::NDRange global_mexp((NUM_LABELS-1)*BATCH_SIZE);
            cl::NDRange global_mmul2(BATCH_SIZE);
            cl::NDRange global_msub(NUM_LABELS*image_size);

            for (int i=0;i<EPOCH;i++) {
                vector<char> tmp_images = data.get_batch_train_images(BATCH_SIZE);
                vector<float> tmp_labels = data.get_batch_train_labels(BATCH_SIZE);

                // device 0
                d_X = cl::Buffer(context, tmp_images.begin(), tmp_images.end(), true);
                d_Y_ = cl::Buffer(context, tmp_labels.begin(), tmp_labels.end(), true);
                // device 1

#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration buffer created" << endl;
#endif
                /*
                 * Y = Wx + b
                 */
                // device 0
                mmul(cl::EnqueueArgs(queue, global_mmul),
                        0, image_size, NUM_LABELS-1, BATCH_SIZE, d_W, d_X, d_Y);
                madd(cl::EnqueueArgs(queue, global_mexp),
                        0, (NUM_LABELS-1)*BATCH_SIZE, d_B, d_Y);
                // device 1
                mmul(cl::EnqueueArgs(queue2, global_mmul),
                        NUM_LABELS-1, image_size, NUM_LABELS, BATCH_SIZE, d_W, d_X, d_Y);
                madd(cl::EnqueueArgs(queue2, global_mexp),
                        (NUM_LABELS-1)*BATCH_SIZE, NUM_LABELS*BATCH_SIZE, d_B, d_Y);
#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration finish Wx+b" << endl;
#endif

                /*
                 * softmax
                 */
                // device 0
                mexp(cl::EnqueueArgs(queue, global_mexp),
                        0, (NUM_LABELS-1)*(BATCH_SIZE), d_Y, d_SUM);
                mnormalize(cl::EnqueueArgs(queue, global_mexp),
                        0, (NUM_LABELS-1)*(BATCH_SIZE), d_Y, d_SUM);
                // device 1
                mexp(cl::EnqueueArgs(queue2, global_mexp),
                        (NUM_LABELS-1)*(BATCH_SIZE), NUM_LABELS*(BATCH_SIZE), d_Y, d_SUM);
                mnormalize(cl::EnqueueArgs(queue2, global_mexp),
                        (NUM_LABELS-1)*(BATCH_SIZE), NUM_LABELS*(BATCH_SIZE), d_Y, d_SUM);
                /*
                mexp2(cl::EnqueueArgs(queue2, global_mexp2),
                        1*BATCH_SIZE, d_Y2, d_SUM2);
                mnormalize2(cl::EnqueueArgs(queue2, global_mexp2),
                        1*BATCH_SIZE, d_Y2, d_SUM2);
                        */
#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration finish softmax" << endl;
#endif

                /*
                 * update weight with gradient descent algorithm
                 *  - its cost function is cross entropy
                 */
                // device 0
                msub(cl::EnqueueArgs(queue, global_mexp),
                        0, (NUM_LABELS-1)*(BATCH_SIZE), 1.0, d_Y_, d_Y);
                // device1
                msub(cl::EnqueueArgs(queue, global_mexp),
                        (NUM_LABELS-1)*(BATCH_SIZE), NUM_LABELS*(BATCH_SIZE), 1.0, d_Y_, d_Y);
#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration mexp finished" << endl;
#endif
                // device 0
                mmul2(cl::EnqueueArgs(queue, global_mmul2),
                        0, BATCH_SIZE, NUM_LABELS-1, image_size, d_Y, d_X, d_W_);
                // device 1
                mmul2(cl::EnqueueArgs(queue2, global_mmul2),
                        NUM_LABELS-1,BATCH_SIZE, NUM_LABELS, image_size, d_Y, d_X, d_W_);
#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration mmul finished" << endl;
#endif
                // device 0
                msub(cl::EnqueueArgs(queue, global_msub),
                        0, (NUM_LABELS-1)*image_size, 0.5, d_W_, d_W);
                // device 1
                msub(cl::EnqueueArgs(queue2, global_msub),
                        (NUM_LABELS-1)*image_size, NUM_LABELS*image_size, 0.5, d_W_, d_W);
#ifdef ITER_DEBUG
                queue.finish();
                cout << i << " Iteration finished" << endl;
#endif
            }

            queue.finish();

            double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            printf("\nTrain ends in %lf seconds\n", rtime);

            int correct_num = 0;
            /*
             * Check accuracy of model
             */
            for (int i=0; i<data.get_test_size()/BATCH_SIZE; i++) {
                vector<char> tmp_images = data.get_batch_test_images(BATCH_SIZE);
                vector<float> tmp_labels = data.get_batch_test_labels(BATCH_SIZE);
                d_X = cl::Buffer(context, tmp_images.begin(), tmp_images.end(), true);
                d_Y_ = cl::Buffer(context, tmp_labels.begin(), tmp_labels.end(), true);
                /*
                 * Y = Wx + b
                 */
                mmul(cl::EnqueueArgs(queue, global_mmul),
                        0, image_size, NUM_LABELS, BATCH_SIZE, d_W, d_X, d_Y);
                madd(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_B, d_Y);

                /*
                 * softmax
                 */
                mexp(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_Y, d_SUM);
                mnormalize(cl::EnqueueArgs(queue, global_mexp),
                        0, NUM_LABELS*BATCH_SIZE, d_Y, d_SUM);

                /*
                 * check whether calculated answer is correct
                 */
                cl::copy(queue, d_Y, h_Y.begin(), h_Y.end());
                for (int j=0; j<BATCH_SIZE; j++) {
                    int idx = j*NUM_LABELS;
                    float max_p = 0.0f;
                    int max_idx = -1;
                    int sol_idx = -1;
                    for (int k=0; k<NUM_LABELS; k++) {
                        if (max_p < h_Y[idx+k]) {
                            max_p = h_Y[idx+k];
                            max_idx = idx+k;
                        } 
                        if (tmp_labels[idx+k] == 1) {
                            sol_idx = idx+k;
                        }
                    }
                    if (sol_idx == max_idx) {
                        correct_num += 1;
                    }
                }
            }

            queue.finish();

            printf("Accuracy : %d / %d\n", correct_num, data.get_test_size());

            double rtime2 = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            printf("\nTest ends in %lf seconds\n", rtime2);


            data.CleanUp();
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
