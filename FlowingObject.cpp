#ifdef _WIN32
#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(disable: 4819)
#endif

#include <iostream>
#include <sstream>
#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "OpticalFlowPyrLK.h"
#include "FlowingObject.h"

int FlowingObject::create_object_id()
{
	static int next_flowing_object_id = 0;

	int id = next_flowing_object_id;
	next_flowing_object_id++;

	return id;
}

FlowingObject::FlowingObject() : id_(create_object_id()), rect_(), disposed_(false)
{
}

FlowingObject::FlowingObject(const cv::Rect &rect) : id_(create_object_id()), rect_(rect), disposed_(false)
{
}

FlowingObject::FlowingObject(const FlowingObject &obj)
{
	copy(obj, *this);
}

FlowingObject& FlowingObject::operator=(const FlowingObject &obj)
{
	copy(obj, *this);
	return *this;
}

void FlowingObject::copy(const FlowingObject &src, FlowingObject &dst)
{
	dst.id_ = src.id_;
	dst.rect_ = src.rect_;
	dst.disposed_ = src.disposed_;
}

FlowingObject::~FlowingObject()
{

}

int FlowingObject::id() const
{
	return id_;
}

bool FlowingObject::is_disposed() const
{
	return disposed_;
}

cv::Point FlowingObject::center() const
{
	cv::Point center;

	center.x = rect_.x + rect_.width / 2;
	center.y = rect_.y + rect_.height / 2;

	return center;
}

bool FlowingObject::is_inner(const cv::Size &size)
{
	cv::Point p = center();

	if (p.x < 0) return false;
	if (p.y < 0) return false;
	if (p.x >= size.width) return false;
	if (p.y >= size.height) return false;

	return true;
}

void FlowingObject::move(const cv::Point2f &diff)
{
	rect_.x += (int)diff.x;
	rect_.y += (int)diff.y;
}

void FlowingObject::process(const OpticalFlowPyrLK &optical_flow)
{
	if (is_disposed()) return;

	cv::Point2f diff = optical_flow.get_flow(this->rect_);
	
	move(diff);

	if (is_inner(optical_flow.frame_size()) == false) {
		disposed_ = true;
	}
}

void FlowingObject::draw(cv::Mat &canvas) const
{
	if (is_disposed()) return;

	cv::rectangle(canvas, rect_, CV_RGB(255, 0, 255), 2);

	std::stringstream ss;
	ss << "obj_id=" << id();

	int w = 2;
	float scale = 0.7f;
	int offset_x = -20;
	int offset_y = -10;

	for (int dy = -w; dy <= w; ++dy) {
		for (int dx = -w; dx <= w; ++dx) {
			cv::putText(canvas, ss.str().c_str(), cv::Point(rect_.x + offset_x + dx, rect_.y + offset_y + dy), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 2, CV_AA);
		}
	}

	cv::putText(canvas, ss.str().c_str(), cv::Point(rect_.x + offset_x, rect_.y + offset_y), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 2, CV_AA);
}

////////////////////////////////////////////////////////////////////////////
FlowingObjectContainer::FlowingObjectContainer()
{

}

FlowingObjectContainer::~FlowingObjectContainer()
{

}

unsigned int FlowingObjectContainer::size() const
{
	return this->flowing_objects_.size();
}

void FlowingObjectContainer::clear()
{
	this->flowing_objects_.clear();
}

int FlowingObjectContainer::append(const cv::Rect &rect)
{
	FlowingObject obj(rect);
	flowing_objects_.push_back(obj);
	return obj.id();
}

bool FlowingObjectContainer::get(const int &object_id, FlowingObject &result)
{
	std::vector<FlowingObject>::iterator it;
	for (it = flowing_objects_.begin(); it != flowing_objects_.end(); ++it) {
		if (it->id() == object_id) {
			result = (*it);
			return true;
		}
	}
	return false;
}

bool is_disposed_(const FlowingObject &obj)
{
	return obj.is_disposed();
}

void FlowingObjectContainer::process(const OpticalFlowPyrLK &optical_flow)
{
	std::vector<FlowingObject>::iterator it;
	for (it = flowing_objects_.begin(); it != flowing_objects_.end(); ++it) {
		it->process(optical_flow);
	}

	// disposedÉtÉâÉOÇ™ïtÇ¢ÇƒÇ¢ÇÈObjectÇçÌèú
	flowing_objects_.erase(std::remove_if(flowing_objects_.begin(), flowing_objects_.end(), is_disposed_), flowing_objects_.end());
}

void FlowingObjectContainer::draw(cv::Mat &canvas) const
{
	std::vector<FlowingObject>::const_iterator it;
	for (it = flowing_objects_.begin(); it != flowing_objects_.end(); ++it) {
		it->draw(canvas);
	}
}