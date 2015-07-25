#pragma once

#include <vector>
#include <opencv2/core.hpp>

class OpticalFlowPyrLK;

class FlowingObject
{
public:
	FlowingObject();
	FlowingObject(const cv::Rect &rect);
	FlowingObject(const FlowingObject &obj);
	FlowingObject& operator=(const FlowingObject &obj);
	static void copy(const FlowingObject &src, FlowingObject &dst);

	virtual ~FlowingObject();

	int id() const;
	bool is_disposed() const;

	cv::Point center() const;
	bool is_inner(const cv::Size &size);

	void move(const cv::Point2f &diff);

	void process(const OpticalFlowPyrLK &optical_flow);
	void draw(cv::Mat &canvas) const;

	static int create_object_id();

protected:
	int id_;
	cv::Rect rect_;
	bool disposed_;
};

class FlowingObjectContainer {
public:
	FlowingObjectContainer();
	virtual ~FlowingObjectContainer();

	unsigned int size() const;

	void clear();
	int append(const cv::Rect &rect);

	bool get(const int &object_id, FlowingObject &result);

	void process(const OpticalFlowPyrLK &optical_flow);
	void draw(cv::Mat &canvas) const;

protected:
	std::vector<FlowingObject> flowing_objects_;
};