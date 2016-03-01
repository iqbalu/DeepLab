// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void BinaryLabelDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void BinaryLabelDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void BinaryLabelDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 
  //LOG(INFO)<<"Binary Dropout Layer";
  Blob<Dtype>* data         = bottom[0];
  const Dtype* bottom_data  = data->cpu_data();
  const int d_count         = data->count();
  const int d_channels      = data->channels();
  const int height          = data->height();
  const int width           = data->width();
  const int dim             = height * width;

  Blob<Dtype>* labels_sum   = bottom[1];
  const Dtype* sum   		= labels_sum->cpu_data();

  Dtype* top_data           = top[0]->mutable_cpu_data();
  unsigned int* mask        = rand_vec_.mutable_cpu_data();

  if (Caffe::phase() == Caffe::TRAIN) {

    CHECK_EQ(bottom[0]->num(), 1) << "Currently only batch size 1 is supported";
    CHECK_EQ(bottom[1]->num(), 1) << "Currently only batch size 1 is supported";
		
    /// calculate probabilities

    int fgcount             = caffe_cpu_asum(labels_sum->count(), sum);
    int bgcount             = labels_sum->count() - fgcount;
    CHECK_GT(fgcount, 0);
    
    // Probability for background pixel to be kept
    Dtype fgProb            = (fgcount / (Dtype)(bgcount + fgcount));
    Dtype bgProb            = 3.25 * fgProb;
    bgProb                  = std::min(bgProb, 1-fgProb);

    threshold_              = 1. - bgProb;
    uint_thres_             = static_cast<unsigned int>(UINT_MAX * threshold_);
    scale_                  = 1.0 / (1. - threshold_+fgProb);
    
    // Create Random numbers
  	caffe_rng_bernoulli(d_count, 1. - threshold_, mask);

   // Dtype* mask_copy = new Dtype[dim]();
    /// apply dropout on convolution channels
    //FIXME: @andreas: try to remove this nested for loop
    int bcount = 0;
    for(int c=0; c<d_channels; c++){
      for(int h=0;h<height;h++){
         for(int w=0;w<width;w++){
           int idx  = ((bcount*d_channels + c)*height + h)*width + w;
			
            // Add GT to Mask
            int sIdx        = h*width + w;
            mask[idx]       = sum[sIdx] == 1 ? 1 : mask[idx];
            
            //mask_copy[sIdx] = sum[sIdx] == 1 ? 1 : mask[idx];
            top_data[idx]  = bottom_data[idx]*mask[idx] * scale_;

            }
        }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void BinaryLabelDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (Caffe::phase() == Caffe::TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}



#ifdef CPU_ONLY
STUB_GPU(BinaryLabelDropoutLayer);
#endif

INSTANTIATE_CLASS(BinaryLabelDropoutLayer);
REGISTER_LAYER_CLASS(BINARY_LABEL_DROPOUT, BinaryLabelDropoutLayer);
}  // namespace caffe
