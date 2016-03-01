#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace caffe {

template <typename Dtype>
void SegAccuracyMLLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    confusion_matrices_.clear();
    confusion_matrices_.resize(bottom[0]->channels());

    SegAccuracyParameter seg_accuracy_param = this->layer_param_.seg_accuracy_param();
    for (int c = 0; c < seg_accuracy_param.ignore_label_size(); ++c){
        ignore_label_.insert(seg_accuracy_param.ignore_label(c));
    }

    num_labels_ = seg_accuracy_param.num_labels();
    for(int c=0;c<bottom[0]->channels();c++)
    {
        confusion_matrices_[c] = new ConfusionMatrix();
        confusion_matrices_[c]->resize(num_labels_); // <---- foreground and background
    }

}

template <typename Dtype>
void SegAccuracyMLLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    CHECK_LE(1, bottom[0]->channels())
                                    << "top_k must be less than or equal to the number of channels (classes).";

    CHECK_EQ(bottom[0]->num(), bottom[1]->num())
                                    << "The data and label should have the same number.";

    CHECK_EQ(bottom[0]->num(), 1)
                                    << "Currently only a batch size of 1 is supported";

    CHECK_EQ(bottom[1]->channels(), bottom[0]->channels())
                                    << "Data channels and label channels must match";

    CHECK_EQ(bottom[0]->height(), bottom[1]->height())
                                    << "The data should have the same height as label.";

    CHECK_EQ(bottom[0]->width(), bottom[1]->width())
                                    << "The data should have the same width as label.";

    top[0]->Reshape(1, 1, 1, 4);
}

template <typename Dtype>
void SegAccuracyMLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data      = bottom[0]->cpu_data();
    const Dtype* bottom_label     = bottom[1]->cpu_data();

    Dtype* top_accuracies         = top[0]->mutable_cpu_data();

    int channels                  = bottom[0]->channels();
    int height                    = bottom[0]->height();
    int width                     = bottom[0]->width();

    int label_channels            = bottom[1]->channels();
    int label_height              = bottom[1]->height();
    int label_width               = bottom[1]->width();

    int data_index, label_index;

    // Initialize accumulated confusion matrix
    ConfusionMatrix accumulated_confusion;
    accumulated_confusion.resize(num_labels_);

    // remove old predictions if reset() flag is true
    if (this->layer_param_.seg_accuracy_param().reset()) {
        LOG(INFO) << "RESET";
        for(int c=0;c<channels;c++)
        {
            confusion_matrices_[c]->clear();
        }
    }
    CHECK_EQ(width, label_width) << "Data width and label width not equal";
    CHECK_EQ(height, label_height) << "Data height and label height not equal";
    CHECK_EQ(channels, label_channels) << "Data channels and label channels not equal";

    int true_fg = 0;

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                data_index = (c * height + h) * width + w;
                label_index = (c * height + h) * width + w;

                int data_label = bottom_data[data_index] < static_cast<Dtype>(0.5) ? static_cast<int>(0) : static_cast<int>(1);
                const int gt_label = static_cast<int>(bottom_label[label_index]);

                if (ignore_label_.count(gt_label) != 0) {
                // ignore the pixel with this gt_label
                    continue;
                }

                confusion_matrices_[c]->accumulate(gt_label, data_label);
            }
        }

        accumulated_confusion.accumulate((*confusion_matrices_[c]));
    }

    /// calculate average accuracies
    /// [0] -> background
    /// [1] -> foreground
    /// [2] -> overall
    LOG(INFO) << "#####SegAccuracyMLLayer####";
    LOG(INFO) << "[0] -> avgPrecision\t [1] -> avgRecall\t [2] -> Accuracy \t[3] -> avgJaccard";
    top_accuracies[0] = static_cast<Dtype>(accumulated_confusion.avgPrecision());
    top_accuracies[1] = static_cast<Dtype>(accumulated_confusion.avgRecall(false));
    top_accuracies[2] = static_cast<Dtype>(accumulated_confusion.accuracy());
    top_accuracies[3] = static_cast<Dtype>(accumulated_confusion.avgJaccard());

    /*LOG(INFO) << "Accuracies of each channel: ";
    for(int c=0;c<14;c++)
    {
        LOG(INFO) << "\t Channel " << c;
        LOG(INFO) << "\t\tBackground: " << static_cast<Dtype>(confusion_matrices_[c]->accuracy(0));
        LOG(INFO) << "\t\tForeground: " << static_cast<Dtype>(confusion_matrices_[c]->accuracy(1));
        LOG(INFO) << "\t\tOverall: " << static_cast<Dtype>(confusion_matrices_[c]->accuracy());
    } */

    /*
    cv::Mat test(cv::Size(width, height), CV_32FC1, bottom[0]->mutable_cpu_data() + 1 * height * width);
    cv::Mat gt(cv::Size(width, height), CV_32FC1, bottom[1]->mutable_cpu_data() + 1 * height * width);
    cv::Mat test_cop = test.clone();
    cv::Mat test_gt = gt.clone();
    cv::resize(test_cop, test_cop, cv::Size(10 * width, 10 * height));
    cv::resize(test_gt, test_gt, cv::Size(10 * width, 10 * height));

    cv::imshow("test", test_cop);
    cv::imshow("gt", test_gt);
    cv::waitKey(10); */
}

INSTANTIATE_CLASS(SegAccuracyMLLayer);
REGISTER_LAYER_CLASS(SEG_ACCURACY_ML, SegAccuracyMLLayer);
}  // namespace caffe
