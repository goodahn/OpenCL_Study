#ifndef MNIST
#define MNIST

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

class MnistData {
    private:
        int m_train_size;
        int m_test_size;
        vector<float> m_train_labels;
        vector<float> m_test_labels;
        vector<char> m_train_images;
        vector<char> m_test_images;
        int m_image_row;
        int m_image_col;
        int m_train_image_pos;
        int m_train_label_pos;
        int m_test_image_pos;
        int m_test_label_pos;
    public:
        MnistData();
        void read_label_and_image(const char* train_labels,
                                  const char* train_images,
                                  const char* test_answers,
                                  const char* test_images);
        int get_train_size(void);
        int get_test_size(void);
        char* get_nth_image(int n);
        vector<char> get_batch_train_images(int size);
        vector<float> get_batch_train_labels(int size);
        vector<char> get_batch_test_images(int size);
        vector<float> get_batch_test_labels(int size);

        int get_image_size(void);
        void CleanUp(void);
};

int ntol(int x);

#endif
