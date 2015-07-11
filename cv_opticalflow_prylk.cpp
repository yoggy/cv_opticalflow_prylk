//
// cv_opticalflow_prylk.cpp - OpenCVを使ったオプティカルフローのサンプル
//
// 参考 :
//     OpenCV3.0/sources/samples/cpp/lkdemo.cpp
//
#ifdef _WIN32
#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(disable: 4819)
typedef UINT64 tick_t;
#ifdef _DEBUG
#pragma comment(lib, "opencv_world300d.lib")
#else
#endif
#pragma comment(lib, "opencv_world300.lib")
#endif

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

double distance(const cv::Point &p0, const cv::Point &p1) {
	int dx = p1.x - p0.x;
	int dy = p1.y - p0.y;
	double dd = (double)(dx * dx + dy * dy);
	double d = (float)std::sqrt(dd);
	return d;
}

tick_t get_now_tick()
{
#ifdef _WIN32
	return ::GetTickCount64();
#endif
}

// オプティカルフロー格納クラス
class OpticalFlow_GoodFeature {
public:
	OpticalFlow_GoodFeature()
		: max_count_(1000), min_count_(300),
		winsize_(31, 31),
		term_criteria_(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
		epsilon_(0.00001),
		feature_life_time_(2000),
		append_interval_(1000),
		max_tracking_length_(100)
	{
	}

	void feature_life_time(const tick_t &val) {
		if (val <= 0) return;
		feature_life_time_ = val;
	}

	void append_interval(const tick_t &val) {
		if (val <= 0) return;
		append_interval_ = val;
	}

	void clear() {
		frame0_.release();
		frame1_.release();
		features0_.clear();
		features1_.clear();
		features_ticks_.clear();
		status_.clear();
		err_.clear();
		last_append_tick_ = 0;
	}

	bool is_tracked(const unsigned int &idx) const {
		if (!status_[idx]) return false;
		if ((get_now_tick() - features_ticks_[idx]) > feature_life_time_) return false;
		if (distance(features0_[idx], features1_[idx]) > max_tracking_length_) return false;
		return true;
	}

	void append_features(const cv::Mat &img) {
		if (img.empty()) return;

		cv::Mat target_img = img;
		if (target_img.type() == CV_8UC3) {
			cv::cvtColor(target_img, target_img, CV_BGR2GRAY);
		}

		std::vector<cv::Point2f> results;
		cv::goodFeaturesToTrack(target_img,results, max_count_, 0.01, 10, cv::Mat(), 3, 0, 0.04);

		features1_.insert(features1_.end(), results.begin(), results.end());

		tick_t t = get_now_tick();
		for (unsigned int i = 0; i < results.size(); ++i) {
			features_ticks_.push_back(t);
		};

		last_append_tick_ = get_now_tick();
	}

	void calc(const cv::Mat &img) {
		cv::Mat target_img = img;
		if (target_img.type() == CV_8UC3) {
			cv::cvtColor(target_img, target_img, CV_BGR2GRAY);
		}

		// トラッキングできている特徴点を前に詰めてリサイズ
		int count = 0;
		for (unsigned int i = 0; i < status_.size(); ++i) {
			if (is_tracked(i)) {
				features1_[count] = features1_[i];
				features_ticks_[count] = features_ticks_[i];
				count++;
			}
		}
		features1_.resize(count);
		features_ticks_.resize(count);

		// 一定時間経過後、または特徴点が少ない場合は特徴点を追加
		if ((int)features1_.size() < min_count_ || (get_now_tick() - last_append_tick_) >= append_interval_) {
			append_features(target_img);
		}

		// 履歴処理
		features0_ = features1_;
		features1_.clear();

		frame1_.copyTo(frame0_);
		frame1_ = target_img;
		if (frame0_.empty()) frame0_ = target_img;

		cv::calcOpticalFlowPyrLK(frame0_, frame1_, features0_, features1_, 
			status_, err_, winsize_, 2, term_criteria_, 0, epsilon_);
	}

	void draw(cv::Mat &canvas) const {
		for (unsigned int i = 0; i < status_.size(); ++i) {
			if (is_tracked(i)) {
				cv::Point p0 = features0_[i];
				cv::Point p1 = features1_[i];

				cv::line(canvas, p0, p1, CV_RGB(0,255,0), 1);
				cv::circle(canvas, p1, 2, CV_RGB(255,0,0), CV_FILLED);
			}
		}
	}

protected:
	int max_count_;
	int min_count_;
	cv::Mat frame0_;
	cv::Mat frame1_;
	std::vector<cv::Point2f> features0_;
	std::vector<cv::Point2f> features1_;
	std::vector<tick_t> features_ticks_;
	std::vector<uchar> status_;
	std::vector<float> err_;

	cv::Size winsize_;
	cv::TermCriteria term_criteria_;
	double epsilon_;

	tick_t feature_life_time_;
	tick_t last_append_tick_;
	tick_t append_interval_;

	int max_tracking_length_;
};

int main(int argc, char* argv[])
{
	OpticalFlow_GoodFeature optical_flow;
	cv::Mat capture_img;
	bool show_capture_img = true;
	tick_t old_t = 0;

	cv::VideoCapture capture;
	if (capture.open(0) == false) {
		printf("error : capture.open() failed...\n");
		return -1;
	}
	capture >> capture_img;

	cv::Mat canvas(capture_img.size(), CV_8UC3);

	while (true) {
		capture >> capture_img;
		optical_flow.calc(capture_img);

		canvas.setTo(0);
		if (show_capture_img) {
			capture_img.copyTo(canvas);
		}
		optical_flow.draw(canvas);

		cv::imshow("cv_opticalflow_prylk", canvas);

		// process key event
		int c = cv::waitKey(1);
		if (c == 'c') {
			optical_flow.clear();
		}
		else if (c == 'd') {
			show_capture_img = !show_capture_img;
		}
		else if (c == ' ') {
			std::cout << "adjust append_interval & feature_life_time..." << std::endl;
			tick_t t = get_now_tick();
			if (old_t > 0) {
				tick_t diff = t - old_t;
				optical_flow.append_interval(diff);
				optical_flow.feature_life_time(diff * 2);
				optical_flow.clear();
			}
			old_t = t;
		}
		else if (c == 27) {
			break;
		}
	}

	return 0;
}

