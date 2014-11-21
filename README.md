cv_opticalflow_prylk.cpp
====
OpenCV���g�����I�v�e�B�J���t���[�̃T���v���B

cv::goodFeaturesToTrack()��cv::calcOpticalFlowPyrLK()���g�p���A
���o���������_�̃I�v�e�B�J���t���[��ǐՁB

<pre>

void append_features(const cv::Mat &gray_img, std::vector&lt;cv::Point2f> &features)
{
    std::vector&lt;cv::Point2f> points;
    cv::goodFeaturesToTrack(gray_img, points, MAX_COUNT, 0.01, 10, cv::Mat(), 3, 0, 0.04);
    features.insert(features.end(), points.begin(), points.end());
}

int main(int argc, char* argv[])
{
    �E
    �E
    �E
    while(true) {
        // 1�t���[���O�̏�Ԃ�ۑ�
        features0 = features1;
        features1.clear();

        frame1_gray.copyTo(frame0_gray);
        capture >> frame1_rgb;
        cv::cvtColor(frame1_rgb, frame1_gray, CV_BGR2GRAY);

        if (features0.size() == 0) {
            append_features(frame1_gray, features1);
        }
        else {
            std::vector&lt;uchar> status;
            std::vector&lt;float> err;

            // �����_�̃I�v�e�B�J���t���[���v�Z
            cv::calcOpticalFlowPyrLK(frame0_gray, frame1_gray, features0, features1,
                status, err, winsize, 3, term_criteria, 0, epsilon);

            // �g���b�L���O�ł��������_�̂ݎc��
            int count = 0;
            for (unsigned int i = 0; i &lt; features1.size(); ++i) {
                if (status[i]) {
                    features1[count++] = features1[i];
                }
            }
            features1.resize(count);

            // �����_�����Ȃ��ꍇ�͒ǉ����Ă���
            if (features1.size() &lt; MIN_COUNT) {
                append_features(frame1_gray, features1);
            }
        }
    �E
    �E
    �E
</pre>
