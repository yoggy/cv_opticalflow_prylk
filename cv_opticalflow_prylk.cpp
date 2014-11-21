//
// cv_opticalflow_prylk.cpp - OpenCVを使ったオプティカルフローのサンプル
//
// 参考 :
//     OpenCV2.4.10/sources/samples/cpp/lkdemo.cpp
//
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <vector>

#ifdef _WIN32
#ifdef _DEBUG
#pragma comment(lib, "opencv_core2410d.lib")
#pragma comment(lib, "opencv_imgproc2410d.lib")
#pragma comment(lib, "opencv_highgui2410d.lib")
#pragma comment(lib, "opencv_video2410d.lib")
#else
#pragma comment(lib, "opencv_core2410.lib")
#pragma comment(lib, "opencv_imgproc2410.lib")
#pragma comment(lib, "opencv_highgui2410.lib")
#pragma comment(lib, "opencv_video2410.lib")
#endif
#endif

#define MAX_COUNT 500
#define MIN_COUNT 300

void append_features(const cv::Mat &gray_img, std::vector<cv::Point2f> &features)
{
	std::vector<cv::Point2f> points;
	cv::goodFeaturesToTrack(gray_img, points, MAX_COUNT, 0.01, 10, cv::Mat(), 3, 0, 0.04);

	features.insert(features.end(), points.begin(), points.end());
}

int main(int argc, char* argv[])
{
	cv::Mat frame0_gray, frame1_gray, frame1_rgb;
	std::vector<cv::Point2f> features0, features1;

	cv::VideoCapture capture;
	if (capture.open(0) == false) {
		printf("error : capture.open() failed...\n");
		return -1;
	}

	cv::Size winsize(31, 31);
	cv::TermCriteria term_criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
	double epsilon = 0.00001;

	while(true) {
		// 1フレーム前の状態を保存
		features0 = features1;
		features1.clear();

		frame1_gray.copyTo(frame0_gray);
		capture >> frame1_rgb;
		cv::cvtColor(frame1_rgb, frame1_gray, CV_BGR2GRAY);

		if (features0.size() == 0) {
			append_features(frame1_gray, features1);
		}
		else {
			std::vector<uchar> status;
			std::vector<float> err;

			// 特徴点のオプティカルフローを計算
			cv::calcOpticalFlowPyrLK(frame0_gray, frame1_gray, features0, features1,
				status, err, winsize, 3, term_criteria, 0, epsilon);

			// トラッキングできた特徴点のみ残す
			int count = 0;
			for (unsigned int i = 0; i < features1.size(); ++i) {
				if (status[i]) {
					features1[count++] = features1[i];
				}
			}
			features1.resize(count);

			// 特徴点が少ない場合は追加しておく
			if (features1.size() < MIN_COUNT) {
				append_features(frame1_gray, features1);
			}
		}

		// 結果の描画
		cv::Mat canvas(frame1_rgb.size(), CV_8UC3);
		frame1_rgb.copyTo(canvas);

		for (unsigned int i = 0; i < features1.size(); ++i) {
			cv::circle(canvas, features1[i], 1, CV_RGB(0,255,0), CV_FILLED);
		}

		cv::imshow("cv_opticalflow_prylk", canvas);

		int c = cv::waitKey(1);
		if (c == ' ') {
			append_features(frame1_gray, features1);
		}
		else if (c == 'c') {
			features1.clear();
		}
		else if (c == 27) {
			break;
		}
	}

	return 0;
}

