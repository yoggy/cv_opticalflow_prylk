cv_opticalflow_prylk.cpp
====
OpenCVを使ったオプティカルフローのサンプル。

cv::goodFeaturesToTrack()とcv::calcOpticalFlowPyrLK()を使用し、
検出した特徴点のオプティカルフローを追跡。

<pre>
class OpticalFlow_GoodFeature {
public:
        ・
        ・
        ・
    void append_features(const cv::Mat &img) {
        if (img.empty()) return;

        cv::Mat target_img = img;
        if (target_img.type() == CV_8UC3) {
            cv::cvtColor(target_img, target_img, CV_BGR2GRAY);
        }

        cv::vector&lt;cv::Point2f> results;
        cv::goodFeaturesToTrack(target_img,results, max_count_, 0.01, 10, cv::Mat(), 3, 0, 0.04);

        features1_.insert(features1_.end(), results.begin(), results.end());
    }
    
    void calc(const cv::Mat &img) {
        ・
        ・
        ・
        // 履歴処理
        features0_ = features1_;
        features1_.clear();

        frame1_.copyTo(frame0_);
        frame1_ = target_img;
        if (frame0_.empty()) frame0_ = target_img;

        cv::calcOpticalFlowPyrLK(frame0_, frame1_, features0_, features1_, 
            status_, err_, winsize_, 3, term_criteria_, 0, epsilon_);
    }
        ・
        ・
        ・
int main(int argc, char* argv[]) {
    OpticalFlow_GoodFeature optical_flow;

    cv::VideoCapture capture;
    capture.open(0);

    while(true) {
        cv::Mat capture_img;
        capture >> capture_img;
        optical_flow.calc(capture_img);
        ・
        ・
        ・
    }
</pre>
