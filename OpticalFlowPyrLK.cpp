#ifdef _WIN32
#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(disable: 4819)
#endif

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "OpticalFlowPyrLK.h"

float length_(const cv::Point2f &v) {
	float ll = (float)(v.x * v.x + v.y * v.y);
	float l = (float)std::sqrt(ll);
	return l;
}

tick_t get_now_tick()
{
#ifdef _WIN32
	return ::GetTickCount64();
#endif
}

OpticalFlowPyrLK::OpticalFlowPyrLK()
	: max_count_(1000), min_count_(300),
	winsize_(31, 31),
	term_criteria_(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
	epsilon_(0.00001),
	feature_life_time_(2000),
	append_interval_(1000),
	max_tracking_length_(50)
{

}

OpticalFlowPyrLK::~OpticalFlowPyrLK()
{

}

void OpticalFlowPyrLK::feature_life_time(const tick_t &val)
{
	if (val <= 0) return;
	this->feature_life_time_ = val;
}

void OpticalFlowPyrLK::append_interval(const tick_t &val)
{
	if (val <= 0) return;
	this->append_interval_ = val;
}

void OpticalFlowPyrLK::clear() {
	frame_old_.release();
	frame_now_.release();
	features_old_.clear();
	features_now_.clear();
	features_diff_.clear();
	features_ticks_.clear();
	status_.clear();
	err_.clear();
	last_append_tick_ = 0;
}

unsigned int OpticalFlowPyrLK::size() const
{
	return features_now_.size();
}

cv::Size OpticalFlowPyrLK::frame_size() const
{
	return frame_now_.size();
}

cv::Point2f OpticalFlowPyrLK::get_flow(const cv::Rect rect) const
{
	cv::Point2f max_diff;
	float max_diff_length = 0.0f;

	if (size() == 0) return cv::Point2f(0.0f, 0.0f);

	for (unsigned i = 0; i < size(); ++i) {
		if (is_tracked(i) == true && is_inner(i, rect) == true) {
			float diff_length = length_(features_diff_[i]);
			if (max_diff_length < diff_length) {
				max_diff = features_diff_[i];
				max_diff_length = diff_length;
			}
		}
	}

	return max_diff;
}

bool OpticalFlowPyrLK::is_inner(const unsigned int &idx, const cv::Rect &rect) const
{
	if (idx < 0 || size() <= idx) return false;
	return rect.contains(features_old_[idx]);
}

bool OpticalFlowPyrLK::is_tracked(const unsigned int &idx) const 
{
	if (!status_[idx]) return false;
	if ((get_now_tick() - features_ticks_[idx]) > feature_life_time_) return false;
	if (length_(features_diff_[idx]) > max_tracking_length_) return false;
	return true;
}

void OpticalFlowPyrLK::append_features(const cv::Mat &img)
{
	if (img.empty()) return;

	cv::Mat target_img = img;
	if (target_img.type() == CV_8UC3) {
		cv::cvtColor(target_img, target_img, CV_BGR2GRAY);
	}

	std::vector<cv::Point2f> results;
	cv::goodFeaturesToTrack(target_img, results, max_count_, 0.01, 10, cv::Mat(), 3, 0, 0.04);

	features_now_.insert(features_now_.end(), results.begin(), results.end());

	tick_t t = get_now_tick();
	for (unsigned int i = 0; i < results.size(); ++i) {
		features_ticks_.push_back(t);
	};

	features_diff_.resize(features_now_.size());

	last_append_tick_ = get_now_tick();
}

void OpticalFlowPyrLK::process(const cv::Mat &img)
{
	cv::Mat target_img = img;
	if (target_img.type() == CV_8UC3) {
		cv::cvtColor(target_img, target_img, CV_BGR2GRAY);
	}

	// トラッキングできている特徴点のみ残す
	int count = 0;
	for (unsigned int i = 0; i < status_.size(); ++i) {
		if (is_tracked(i)) {
			features_now_[count] = features_now_[i];
			features_ticks_[count] = features_ticks_[i];
			count++;
		}
	}
	features_now_.resize(count);
	features_diff_.resize(count);
	features_ticks_.resize(count);

	// 特徴点が少ない場合は追加しておく
	if ((int)features_now_.size() < min_count_ || (get_now_tick() - last_append_tick_) >= append_interval_) {
		append_features(target_img);
	}

	// 新データを旧データにローテーションする
	features_old_ = features_now_;
	features_now_.clear();

	frame_now_.copyTo(frame_old_);
	frame_now_ = target_img;
	if (frame_old_.empty()) frame_old_ = target_img;

	// 疎なオプティカルフローの計算
	cv::calcOpticalFlowPyrLK(frame_old_, frame_now_, features_old_, features_now_,
		status_, err_, winsize_, 2, term_criteria_, 0, epsilon_);

	// 移動距離を計算しておく
	for (unsigned int i = 0; i < features_now_.size(); ++i) {
		features_diff_[i].x = features_now_[i].x - features_old_[i].x;
		features_diff_[i].y = features_now_[i].y - features_old_[i].y;
	}
}

void OpticalFlowPyrLK::draw(cv::Mat &canvas) const
{
	for (unsigned int i = 0; i < status_.size(); ++i) {
		if (is_tracked(i)) {
			cv::Point p0 = features_old_[i];
			cv::Point p1 = features_now_[i];

			cv::line(canvas, p0, p1, CV_RGB(0, 255, 0), 1);
			cv::circle(canvas, p1, 2, CV_RGB(255, 0, 0), CV_FILLED);
		}
	}
}
