#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {


template <typename Dtype>
__global__ void BinaryLabelDropoutForward(const int n, const int chnDim, const Dtype* in,
    unsigned int* mask, const Dtype* sum, const unsigned int threshold, const float scale,
    Dtype* out) {
    int counter = 0;
  CUDA_KERNEL_LOOP(index, n) {

         counter = index % chnDim;
         mask[index] = sum[counter] > 0. ? UINT_MAX : mask[index];
         out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void BinaryLabelDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                      const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> *data             = bottom[0];
  const Dtype* bottom_data      = data->gpu_data();
  const int d_count             = data->count();
  const int d_channels          = data->channels();
  const int height              = data->height();
  const int width               = data->width();
  const int dim                 = height * width;

  Blob<Dtype> *label_sum        = bottom[1];
  const Dtype *sum              = label_sum->gpu_data();

  Dtype* top_data               = top[0]->mutable_gpu_data();
  unsigned int *mask            = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());

  if (Caffe::phase() == Caffe::TRAIN) {
    CHECK_EQ(data->num(), 1) << "Currently only batch size 1 is supported";
    CHECK_EQ(label_sum->num(), 1) << "Currently only batch size 1 is supported";

    /// calculate probabilities

    Dtype fgcount_f; int fgcount;
    caffe_gpu_asum(label_sum->count(), sum, &fgcount_f);

    fgcount = static_cast<int>(fgcount_f);
    int bgcount = label_sum->count() - fgcount;
    CHECK_GT(fgcount,0);

    // Probability for background pixel to be kept
    Dtype fgProb = (fgcount / (Dtype) (bgcount + fgcount));
    Dtype bgProb = 3.25 * fgProb;
    bgProb = std::min(bgProb, 1-fgProb);

    threshold_ = 1. - bgProb;
    uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
    scale_ = 1. / (1. - threshold_ + fgProb);

    // create random numbers
    caffe_gpu_rng_uniform(d_count, mask);

    /// Apply dropout
    BinaryLabelDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(d_count), CAFFE_CUDA_NUM_THREADS>>>(
        d_count, dim, bottom_data, mask, sum, uint_thres_, scale_, top_data);
    CUDA_POST_KERNEL_CHECK;


  } else {
    caffe_copy(d_count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void BinaryLabelDropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void BinaryLabelDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (Caffe::phase() == Caffe::TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      BinaryLabelDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryLabelDropoutLayer);


}  // namespace caffe
