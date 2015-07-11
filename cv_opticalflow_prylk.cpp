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
#ifdef _DEBUG
#pragma comment(lib, "opencv_world300d.lib")
#else
#endif
#pragma comment(lib, "opencv_world300.lib")
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

// オプティカルフロー格納クラス
class OpticalFlow_GoodFeature {
public:
	OpticalFlow_GoodFeature() 
		: max_count_(1000), min_count_(300), 
		winsize_(31, 31),  
		term_criteria_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03), 
		epsilon_(0.00001)
	{
	}

	void clear() {
		frame0_.release();
		frame1_.release();
		features0_.clear();
		features1_.clear();
		status_.clear();
		err_.clear();
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
	}

	void calc(const cv::Mat &img) {
		cv::Mat target_img = img;
		if (target_img.type() == CV_8UC3) {
			cv::cvtColor(target_img, target_img, CV_BGR2GRAY);
		}

		// トラッキングできた特徴点を前に詰めてリサイズ
		int count = 0;
		for (unsigned int i = 0; i < status_.size(); ++i) {
			if (status_[i]) {
				features1_[count++] = features1_[i];
			}
		}
		features1_.resize(count);

		// 特徴点が少ない場合は追加しておく
		if ((int)features1_.size() < min_count_) {
			append_features(target_img);
		}

		// 履歴処理
		features0_ = features1_;
		features1_.clear();

		frame1_.copyTo(frame0_);
		frame1_ = target_img;
		if (frame0_.empty()) frame0_ = target_img;

		cv::calcOpticalFlowPyrLK(frame0_, frame1_, features0_, features1_, 
			status_, err_, winsize_, 3, term_criteria_, 0, epsilon_);
	}

	void draw(cv::Mat &canvas) const {
		for (unsigned int i = 0; i < status_.size(); ++i) {
			if (status_[i]) {
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
	std::vector<uchar> status_;
	std::vector<float> err_;

	cv::Size winsize_;
	cv::TermCriteria term_criteria_;
	double epsilon_;
};

int main(int argc, char* argv[])
{
	OpticalFlow_GoodFeature optical_flow;
	cv::Mat capture_img;

	cv::VideoCapture capture;
	if (capture.open(0) == false) {
		printf("error : capture.open() failed...\n");
		return -1;
	}

	while(true) {
		capture >> capture_img;
		optical_flow.calc(capture_img);

		cv::Mat canvas;
		capture_img.copyTo(canvas);
		optical_flow.draw(canvas);

		cv::imshow("cv_opticalflow_prylk", canvas);

		int c = cv::waitKey(1);
		if (c == ' ') {
			optical_flow.append_features(capture_img);
		}
		else if (c == 'c') {
			optical_flow.clear();
		}
		else if (c == 27) {
			break;
		}
	}

	return 0;
}

