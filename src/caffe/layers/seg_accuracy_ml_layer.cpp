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

// TODO: check we should clear confusion_matrix somewhere!

namespace caffe {

template <typename Dtype>
void SegAccuracyMLLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  confusion_matrix_.clear();
  confusion_matrix_.resize(bottom[0]->channels());
  SegAccuracyParameter seg_accuracy_param = this->layer_param_.seg_accuracy_param();
  for (int c = 0; c < seg_accuracy_param.ignore_label_size(); ++c){
    ignore_label_.insert(seg_accuracy_param.ignore_label(c));
  }
}

template <typename Dtype>
void SegAccuracyMLLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_LE(1, bottom[0]->channels())
      << "top_k must be less than or equal to the number of channels (classes).";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";

  // CHANGE 18.1
 // CHECK_EQ(bottom[1]->channels(), 1)
 //   << "The label should have one channel.";
  CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data should have the same width as label.";

  top[0]->Reshape(1, bottom[0]->channels(), 1, 3);
  //top[0]->Reshape(1, 1, 1, 3);
}

template <typename Dtype>
void SegAccuracyMLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int label_channels = bottom[1]->channels();
  int label_height = bottom[1]->height();
  int label_width = bottom[1]->width();

  //CHECK(false) << "DEBUG: num = " << num << " dim = " << dim << " channels = " << channels << " height = " << height << " width = " << width << " label channels = " << label_channels << " label width = " << label_width << " label height = " << label_height;
  int data_index, label_index;

  int top_k = 1;  // only support for top_k = 1

  // remove old predictions if reset() flag is true
  //if (this->layer_param_.seg_accuracy_param().reset()) {
    confusion_matrix_.clear();
  //}

  CHECK_EQ(width, label_width) << "Data width and label width not equal";
  CHECK_EQ(height, label_height) << "Data height and label height not equal";
  CHECK_EQ(channels, label_channels) << "Data channels and label channels not equal";

  for (int c = 0; c < channels; ++c) {
  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {

          data_index = ((i * channels + c) * height + h) * width + w;
          label_index = ((i * channels + c) * height + h) * width + w;

          int data_label = bottom_data[data_index] < static_cast<Dtype>(0.5) ? static_cast<int>(0) : static_cast<int>(1);
          const int gt_label = static_cast<int>(bottom_label[label_index]);

          if (ignore_label_.count(gt_label) != 0) {
            // ignore the pixel with this gt_label
            continue;
          }

          confusion_matrix_.accumulate(gt_label, data_label);
        }
      }
    }
   //bottom_data  += bottom[0]->offset(1);
   // bottom_label += bottom[1]->offset(1);

  // we report all the resuls for each channel
  top_data[c * 3 + 0] = (Dtype)confusion_matrix_.accuracy();
  top_data[c * 3 + 1] = (Dtype)confusion_matrix_.avgRecall(false);
  top_data[c * 3 + 2] = (Dtype)confusion_matrix_.avgJaccard();

  //top[0]->mutable_cpu_data()[0] = (Dtype)confusion_matrix_.accuracy();
  //top[0]->mutable_cpu_data()[1] = (Dtype)confusion_matrix_.avgRecall(false);
  //top[0]->mutable_cpu_data()[2] = (Dtype)confusion_matrix_.avgJaccard();

    // clear confusion matrix for every channel
    confusion_matrix_.clear();
  }



  /*
  Dtype result;

  switch (this->layer_param_.seg_accuracy_param().metric()) {
  case SegAccuracyParameter_AccuracyMetric_PixelAccuracy:
    result = (Dtype)confusion_matrix_.accuracy();
    break;
  case SegAccuracyParameter_AccuracyMetric_ClassAccuracy:
    result = (Dtype)confusion_matrix_.avgRecall();
    break;
  case SegAccuracyParameter_AccuracyMetric_PixelIOU:
    result = (Dtype)confusion_matrix_.avgJaccard();
    break;
  default:
    LOG(FATAL) << "Unknown Segment accuracy metric.";
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = result;
  // Accuracy layer should not be used as a loss function.
  */
}

INSTANTIATE_CLASS(SegAccuracyMLLayer);
REGISTER_LAYER_CLASS(SEG_ACCURACY_ML, SegAccuracyMLLayer);
}  // namespace caffe
