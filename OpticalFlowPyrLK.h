#pragma once

#ifdef _WIN32
typedef UINT64 tick_t;
#endif

#include <vector>
#include <opencv2/core.hpp>

tick_t get_now_tick();

class OpticalFlowPyrLK {
public:
	OpticalFlowPyrLK();
	virtual ~OpticalFlowPyrLK();

	void feature_life_time(const tick_t &val);
	void append_interval(const tick_t &val);

	void clear();
	unsigned int size() const;

	cv::Size frame_size() const;
	cv::Point2f get_flow(const cv::Rect rect) const;

	bool is_inner(const unsigned int &idx, const cv::Rect &rect) const;

	bool is_tracked(const unsigned int &idx) const;
	void append_features(const cv::Mat &img);

	void process(const cv::Mat &img);
	void draw(cv::Mat &canvas) const;

protected:
	int max_count_;
	int min_count_;
	cv::Mat frame_old_;
	cv::Mat frame_now_;
	std::vector<cv::Point2f> features_old_;
	std::vector<cv::Point2f> features_now_;
	std::vector<tick_t> features_ticks_;
	std::vector<cv::Point2f> features_diff_;
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


