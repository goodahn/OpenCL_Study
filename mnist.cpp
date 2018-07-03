#include "mnist.hpp"
using namespace std;

MnistData::MnistData()
{
    m_train_size = 0;
    m_test_size = 0;
    m_train_labels = vector<float>();
    m_train_images = vector<char>();
    m_test_labels = vector<float>();
    m_test_images = vector<char>();
    m_image_row = 0;
    m_image_col = 0;
    m_train_image_pos = 0;
    m_train_label_pos = 0;
    m_test_image_pos = 0;
    m_test_label_pos = 0;
}

/*
 * Read data from mnist data.
 */
void MnistData::read_label_and_image(const char* train_labels,
                          const char* train_images,
                          const char* test_labels,
                          const char* test_images)
{
    ifstream f_train_labels;
    ifstream f_test_labels;
    ifstream f_train_images;
    ifstream f_test_images;

    int int_data;
    char char_data;

    /*
     * Extract train label data from mnist label file.
     */
    f_train_labels.open(train_labels);
    f_train_labels.read((char*)&int_data, 4); int_data = ntol(int_data);
    if (int_data != 2049) {
        cout << "Magic number of train label file is corrupted! " << int_data
             << " is set, instead of 2049" << endl;
        f_train_labels.close();
        f_test_labels.close();
        f_train_images.close();
        return;
    }

    f_train_labels.read((char*)&int_data, 4); int_data = ntol(int_data);
    m_train_size = int_data;
    cout << "There are " << int_data << " train images" << endl;

    for (int i=0;i<int_data;i++) {
        f_train_labels.read(&char_data, 1);
        for (int j=0;j<10;j++) {
            if (j==(int)char_data)
                m_train_labels.push_back((float)1);
            else
                m_train_labels.push_back((float)0);
        }
    }
    cout << "Complete to extract train label data" << endl;
    f_train_labels.close();

    /*
     * Extract train image data from mnist image file.
     */
    f_train_images.open(train_images);
    f_train_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    if (int_data != 2051) {
        cout << "Magic number of train image file is corrupted! " << int_data
             << " is set, instead of 2051 " << endl;
        f_train_images.close();
        return;
    }

    f_train_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    if (m_train_size != int_data) {
        cout << "Number of train labels and number of train image files are different!" << endl;
        cout << m_train_size << "\t\t" << int_data << endl;
        f_train_images.close();
        return;
    }
    f_train_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    m_image_row = int_data;
    f_train_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    m_image_col = int_data;

    int image_size = m_image_row*m_image_col;
    for (int i=0;i<m_train_size;i++) {
        for (int j=0;j<image_size;j++) {
            f_train_images.read(&char_data, 1);
            m_train_images.push_back(char_data);
        }
    }
    f_train_images.close();
    cout << "Complete to extract train image data" << endl;

    /*
     * Extract test label data from mnist answer file.
     */
    f_test_labels.open(test_labels);
    f_test_labels.read((char*)&int_data, 4); int_data = ntol(int_data);
    if (int_data != 2049) {
        cout << "Magic number of test label file is corrupted! " << int_data
             << " is set, instead of 2049" << endl;
        f_test_labels.close();
        return;
    }

    f_test_labels.read((char*)&int_data, 4); int_data = ntol(int_data);
    m_test_size = int_data;
    cout << "There are " << int_data << " images" << endl;

    for (int i=0;i<int_data;i++) {
        f_test_labels.read(&char_data, 1);
        for (int j=0;j<10;j++) {
            if (j==(int)char_data)
                m_test_labels.push_back((float)1);
            else
                m_test_labels.push_back((float)0);
        }
    }
    cout << "Complete to extract test label data" << endl;
    f_test_labels.close();

    /*
     * Extract test image data from mnist image file.
     */
    f_test_images.open(test_images);
    f_test_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    if (int_data != 2051) {
        cout << "Magic number of test image file is corrupted! " << int_data
             << " is set, instead of 2051 " << endl;
        f_test_images.close();
        return;
    }

    f_test_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    if (m_test_size != int_data) {
        cout << "Number of test labels and number of test image files are different!" << endl;
        cout << m_test_size << "\t\t" << int_data << endl;
        f_test_images.close();
        return;
    }
    f_test_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    m_image_row = int_data;
    f_test_images.read((char*)&int_data, 4); int_data = ntol(int_data);
    m_image_col = int_data;

    image_size = m_image_row*m_image_col;
    for (int i=0;i<m_test_size;i++) {
        for (int j=0;j<image_size;j++) {
            f_test_images.read(&char_data, 1);
            m_test_images.push_back(char_data);
        }
    }
    f_test_images.close();
    cout << "Complete to extract test image data" << endl;
}

int MnistData::get_train_size(void)
{
    return m_train_size;
}

int MnistData::get_test_size(void)
{
    return m_test_size;
}

int MnistData::get_image_size(void)
{
    return m_image_row*m_image_col;
}

vector<char> MnistData::get_batch_train_images(int size)
{
    int cur_size = m_image_row*m_image_col*size;
    vector<char>::const_iterator begin = m_train_images.begin()+m_train_image_pos;
    vector<char>::const_iterator end = m_train_images.begin()+m_train_image_pos+cur_size;
    vector<char> result(begin, end);
    m_train_image_pos += cur_size;
    return result;
}

vector<float> MnistData::get_batch_train_labels(int size)
{
    vector<float>::const_iterator begin = m_train_labels.begin()+m_train_label_pos;
    vector<float>::const_iterator end = m_train_labels.begin()+m_train_label_pos+size*10;
    vector<float> result(begin, end);
    m_train_label_pos += size*10;
    return result;
}

vector<char> MnistData::get_batch_test_images(int size)
{
    int cur_size = m_image_row*m_image_col*size;
    vector<char>::const_iterator begin = m_test_images.begin()+m_test_image_pos;
    vector<char>::const_iterator end = m_test_images.begin()+m_test_image_pos+cur_size;
    vector<char> result(begin, end);
    m_test_image_pos += cur_size;
    return result;
}

vector<float> MnistData::get_batch_test_labels(int size)
{
    vector<float>::const_iterator begin = m_test_labels.begin()+m_test_label_pos;
    vector<float>::const_iterator end = m_test_labels.begin()+m_test_label_pos+size;
    vector<float> result(begin, end);
    m_test_label_pos += size;
    return result;
}

void MnistData::CleanUp(void)
{
    m_train_labels.clear();
    m_train_images.clear();
    m_test_labels.clear();
    m_test_images.clear();
}

int ntol(int x)
{
    return ((x >> 24) & 0xff) + (((x >> 16) & 0xff) << 8) + (((x >> 8) & 0xff) << 16) + ((x & 0xff) << 24);
}
