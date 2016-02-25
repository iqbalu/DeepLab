#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "matio.h"

namespace caffe {

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) <<
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    if (label_type != ImageDataParameter_LabelType_NONE) {
      iss >> segfn;
    }
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  // CHANGE 18.1
  // If LabelType ist MATPIXEL, read the corresponding segmentation file and obtain the number of segmentation channels
  int label_channels = 1;
  if(label_type == ImageDataParameter_LabelType_MATPIXEL)
  {
      string matPath = root_folder + lines_[lines_id_].second;
      cv::Mat segImg = ReadMatFileToCVMat(matPath, new_height, new_width, "seg_mask");
      CHECK(segImg.data) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
      label_channels = segImg.channels();
  }
  // ENDCHANGE 18.1

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);

    //label

    top[1]->Reshape(batch_size, label_channels, crop_size, crop_size);  // change 18.1
    this->prefetch_label_.Reshape(batch_size, label_channels, crop_size, crop_size); //change 18.1
    this->transformed_label_.Reshape(1, label_channels, crop_size, crop_size);  // change 18.1


  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    //label
    top[1]->Reshape(batch_size, label_channels, height, width);     // change 18.1
    this->prefetch_label_.Reshape(batch_size, label_channels, height, width); // change 18.1
    this->transformed_label_.Reshape(1, label_channels, height, width); // change 18.1
  }

  // image dimensions, for each image, stores (img_height, img_width)
  top[2]->Reshape(batch_size, 1, 1, 2);
  this->prefetch_data_dim_.Reshape(batch_size, 1, 1, 2);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
  // image_dim
  LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
	    << top[2]->channels() << "," << top[2]->height() << ","
	    << top[2]->width();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageSegDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  ImageDataParameter image_data_param    = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const int label_channels = this->transformed_label_.channels();       // CHANGED
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  const int lines_size = lines_.size();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);

  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_.InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
  this->transformed_label_.Reshape(top_shape[0], label_channels, top_shape[2], top_shape[3]);   // CHANGED

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  this->prefetch_data_.Reshape(batch_size, top_shape[1], top_shape[2], top_shape[3]);
  this->prefetch_label_.Reshape(batch_size, label_channels, top_shape[2], top_shape[3]);    // CHANGED

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    std::vector<cv::Mat> cv_img_seg;
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    int img_row, img_col;

    // 18.1. ACCESS TO DATA DIM!!
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      		new_height, new_width, is_color, &img_row, &img_col);

    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    cv_img_seg.push_back(cv_img);
    double min_val, max_val;
    cv::minMaxLoc(cv_img_seg[0], &min_val, &max_val);

    if (label_type == ImageDataParameter_LabelType_PIXEL) {
      cv::Mat cv_seg = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                  new_height, new_width, false);
      CHECK(cv_seg.data) << "Could not load " << lines_[lines_id_].second;
      cv_img_seg.push_back(cv_seg);
    } else if(label_type == ImageDataParameter_LabelType_MATPIXEL)
    {
      // LABEL TYPE MATPIXEL
      // converting .mat to cv::Mat
      string matPath = root_folder + lines_[lines_id_].second;
      cv::Mat segImg = ReadMatFileToCVMat(matPath, new_height, new_width, "seg_mask");
      cv_img_seg.push_back(segImg);
      if (!cv_img_seg[1].data) {
           DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
      }
    }
    else if (label_type == ImageDataParameter_LabelType_IMAGE) {
      const int label = atoi(lines_[lines_id_].second.c_str());
      cv::Mat cv_seg(cv_img_seg[0].rows, cv_img_seg[0].cols,
          CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(cv_seg);
    }
    else {
      cv::Mat cv_seg(cv_img_seg[0].rows, cv_img_seg[0].cols,
        CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(cv_seg);
    }

    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply transformations (mirror, crop...) to the image
    int offset;

    offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);

    offset = this->prefetch_label_.offset(item_id);
    this->transformed_label_.set_cpu_data(prefetch_label + offset);

    bool ML = label_type == ImageDataParameter_LabelType_MATPIXEL;

    this->data_transformer_.TransformImgAndSeg(cv_img_seg,
     &(this->transformed_data_), &(this->transformed_label_),
     ignore_label, ML);

    const Dtype* data = this->transformed_data_.cpu_data();
    int cn = this->transformed_data_.count();
    for(int i=0; i<cn; i++){
      CHECK_GE(data[i], -255) << i;
      CHECK_LE(data[i], 255) << i;
      CHECK_EQ(data[i], data[i]);
    }

    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(IMAGE_SEG_DATA, ImageSegDataLayer);
}  // namespace caffe
