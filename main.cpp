#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const cv::String picPath = R"(pics\picture1.jpg)";
const cv::String bookPath = R"(pics\book.jpg)";

class Histogram1D {
private:
    int histSize[1];
    float hranges[2];
    float *ranges[1];
    int channels[1];
public:
    Histogram1D() {

        histSize[0] = 256;
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        ranges[0] = hranges;
        channels[0] = 0;
    }

    Mat getHistogram(const Mat &image) {
        Mat hist;

        calcHist(&image, 1, channels, Mat(), hist, 1, histSize, 0);

        return hist;
    }

    Mat getHistogramImage(const Mat &image, int zoom = 1) {

        Mat hist = getHistogram(image);
        return getImageOfHistogram(hist, zoom);
    }

    static Mat getImageOfHistogram(const Mat &hist, int zoom) {
        double maxVal = 0;
        double minVal = 0;
        minMaxLoc(hist, &minVal, &maxVal, 0, 0);
        int histSize = hist.rows;
        Mat histImg(histSize * zoom, histSize * zoom,
                    CV_8U, Scalar(255));
        int hpt = static_cast<int>(0.9 * histSize);
        for (int h = 0; h < histSize; h++) {
            float binVal = hist.at<float>(h);

            if (binVal > 0) {
                int intensity = static_cast<int>(binVal * hpt / maxVal);
                line(histImg, Point(h * zoom, histSize * zoom),
                     Point(h * zoom, (histSize - intensity) * zoom),
                     Scalar(0), zoom);
            }

        }
        return histImg;

    }

    static Mat applyLookUp(
            const Mat &image,
            const Mat &lookup) {
        Mat result;

        LUT(image, lookup, result);
        return result;
    }
};

class ColorHistogram {
private:
    int histSize[3];
    float hranges[2];
    const float *ranges[3];
    int channels[3];
public:
    ColorHistogram() {
        histSize[0] = histSize[1] = histSize[2] = 256;
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;
    }

    cv::Mat getHistogram(const cv::Mat &image) {
        cv::Mat hist;
        cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
        return hist;
    }
};

void colorReduce(cv::Mat &image, int div = 64) {
    cv::Mat lookup(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
        lookup.at<uchar>(i) = i / div * div + div / 2;
    cv::LUT(image, lookup, image);
}

class ContentFinder {
private:
    float hranges[2];
    const float *ranges[3];
    int channels[3];
    float threshold;
    cv::Mat histogram;
public:
    ContentFinder() : threshold(0.1f) {
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
    }

    void setHistogram(const cv::Mat &h) {
        histogram = h;
        cv::normalize(histogram, histogram, 1.0);
    }

    cv::Mat find(const cv::Mat &image) {
        cv::Mat result;
        hranges[0] = 0.0;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;
        return find(image, hranges[0], hranges[1], channels);
    }

    cv::Mat find(const cv::Mat &image, float minValue, float maxValue, int *channels) {
        cv::Mat result;
        hranges[0] = minValue;
        hranges[1] = maxValue;
        for (int i = 0; i < histogram.dims; i++)
            this->channels[i] = channels[i];
        cv::calcBackProject(&image, 1, channels, histogram, result, ranges, 55.0);
    }
};

template<typename T, int N>
class IntegralImage {
    cv::Mat integralImage;
public:
    IntegralImage(cv::Mat image) {
        cv::integral(image, integralImage,
                     cv::DataType<T>::type);
    }

    cv::Vec<T, N> operator()(int xo, int yo, int width, int height) {
        return (integralImage.at<cv::Vec<T, N>>(yo + height, xo + width) -
                integralImage.at<cv::Vec<T, N>>(yo + height, xo) - integralImage.at<cv::Vec<T, N>>(yo, xo + width) +
                integralImage.at<cv::Vec<T, N>>(yo, xo));
    }
};

int main() {
    Mat result;
    Mat hist;
    Histogram1D h;
    Histogram1D negative;
    Mat hsv;

    Mat image = imread(picPath);
    Mat imageGrey = imread(picPath);
    Mat imageBook = imread(bookPath);
    Mat imageClone = image.clone();
    Mat imageClone2 = image.clone();
    Mat imageColor = image.clone();

    colorReduce(image);
    namedWindow("Image 1-5%", WINDOW_AUTOSIZE);
    imshow("Image 1-5%", image);

    Mat histo = h.getHistogram(image);
    imshow("Histogram 1-5%", h.getHistogramImage(image));

    Rect rect(330, 80, 230, 200);
    Mat imageROI = imageClone(rect);
    int minSat = 65;
    ColorHistogram hc;
    ContentFinder finder;
    imageROI = imageColor(cv::Rect(330, 80, 230, 200));
    cv::Mat shist = hc.getHistogram(imageROI);

    namedWindow(" Imageroi", WINDOW_AUTOSIZE);
    imshow(" Imageroi", imageROI);

    for (int i = 0; i < 256; i++) {
        cout << "Value " << i << " = " << histo.at<float>(i) << endl;
    }

    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = 255 - i;
    }

    LUT(imageROI, lut, result);

    imshow("HistogramNegative", negative.getHistogramImage(result));
    imshow("negative image", result);

    Mat binaryFixed;
    threshold(imageBook, binaryFixed, 70, 255, THRESH_BINARY);
    imshow("binaryFixed", binaryFixed);
    Canny(imageBook, binaryFixed, 10, 100, 3);

    namedWindow(" ImageBook", WINDOW_AUTOSIZE);
    imshow(" ImageBook", binaryFixed);
    Rect Rec(330, 80, 230, 200);
    rectangle(imageClone2, Rec, Scalar(255), 1, 8, 0);
    Mat Roi = imageClone2(Rec);
    Rect WhereRec(0, 0, Roi.cols, Roi.rows);
    Roi.copyTo(imageClone2(WhereRec));

    imshow(" Final result", imageClone2);

    waitKey(0);
    return 0;

}
