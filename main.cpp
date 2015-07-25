//
// cv_opticalflow_prylk.cpp - OpenCVを使ったオプティカルフローのサンプル
//
// 参考 :
//     https://github.com/Itseez/opencv/blob/master/samples/cpp/lkdemo.cpp
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

#include <iostream>
#include <opencv2/highgui.hpp>
#include "OpticalFlowPyrLK.h"
#include "FlowingObject.h"

// draw mode
#define DRAW_CAPTURE_IMG     1
#define DRAW_FEATURE_POINTS  2

int draw_mode_ = DRAW_CAPTURE_IMG | DRAW_FEATURE_POINTS;
#define is_enable(x) ((draw_mode_ & x) != 0)
#define change_draw_mode() {draw_mode_ = (draw_mode_ == 3 ? 1 : draw_mode_+1);}

const char *window_name = "cv_opticalflow_prylk";

OpticalFlowPyrLK optical_flow;
FlowingObjectContainer flowing_objects;

void on_mouse(int event, int x, int y, int, void* userdata)
{
	if (event & CV_EVENT_LBUTTONDOWN) {
		cv::Point p = cv::Point(x, y);

		int size = 60;

		cv::Rect r;
		r.x = p.x - size / 2;
		r.y = p.y - size / 2;
		r.width = size;
		r.height = size;

		flowing_objects.append(r);
	}
}

bool process_key_event()
{
	bool rv = true;
	static tick_t old_t = 0;
	tick_t t;

	int c = cv::waitKey(1);

	switch (c) {
	case 'c':
		optical_flow.clear();
		flowing_objects.clear();
		break;
	case 'd':
		change_draw_mode();
		break;
	case ' ':
		std::cout << "adjust append_interval & feature_life_time..." << std::endl;
		t = get_now_tick();
		if (old_t > 0) {
			tick_t diff = t - old_t;
			optical_flow.append_interval(diff);
			optical_flow.feature_life_time(diff * 2);
			optical_flow.clear();
		}
		old_t = t;
		break;
	case 27:
		rv = false; // exit main loop
		break;
	}
	return rv;
}

int main(int argc, char* argv[])
{
	cv::Mat capture_img;

	cv::VideoCapture capture;
	if (capture.open(0) == false) {
		printf("error : capture.open() failed...\n");
		return -1;
	}
	capture >> capture_img;

	cv::Mat canvas(capture_img.size(), CV_8UC3);

	cv::namedWindow(window_name);
	cv::setMouseCallback(window_name, on_mouse);

	while (true) {
		// capture image 
		capture >> capture_img;

		// process 
		optical_flow.process(capture_img);
		flowing_objects.process(optical_flow);

		// draw image
		canvas.setTo(0);
		if (is_enable(DRAW_CAPTURE_IMG)) {
			capture_img.copyTo(canvas);
		}

		if (is_enable(DRAW_FEATURE_POINTS)) {
			optical_flow.draw(canvas);
		}

		flowing_objects.draw(canvas);

		cv::imshow(window_name, canvas);

		// for key event
		if (process_key_event() == false) break;
	}

	capture.release();
	cv::destroyAllWindows();

	return 0;
}
