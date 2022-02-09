#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const cv::String picPath = R"(pics\hot-charcoal.jpg)";
const cv::String bookPath = R"(pics\book.jpg)";

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

    void setThreshold(float _threshold) {
        threshold = _threshold;
    }

    cv::Mat find(const cv::Mat &image) {
        cv::Mat result;
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;
        return find(image, hranges[0], hranges[1], channels);
    }

    cv::Mat find(const cv::Mat &image, float minValue, float maxValue, int *channels) {
        cv::Mat result;
        hranges[0] = minValue;
        hranges[1] = maxValue;

        for (int i = 0; i < histogram.dims; i++) {
            this->channels[i] = channels[i];
        }

        cv::calcBackProject(&image, 1, channels, histogram, result, ranges, 55.0);

        if (threshold > 0.0) {
            cv::threshold(result, result, 255.0 * threshold, 255.0, cv::THRESH_BINARY);
        }

        return result;
    }
};

void colorReduce(cv::Mat &image, int div = 64) {
    cv::Mat lookup(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
        lookup.at<uchar>(i) = i / div * div + div / 2;
    cv::LUT(image, lookup, image);
}

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

    void setSize(int hintSize) {
        histSize[0] = histSize[1] = histSize[2] = hintSize;
    }

};

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

        calcHist(&image, 1, channels, Mat(), hist, 1, histSize, nullptr);

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
        Mat histImg(histSize * zoom, histSize * zoom, CV_8U, Scalar(255));
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

int main() {
    // 1. Завантажити зображення
    Mat imageBW = imread(picPath, 0);
    namedWindow("Original BW Image", WINDOW_AUTOSIZE);
    imshow("Original BW Image", imageBW);

    Mat imageRGB = imread(picPath);
    namedWindow("Original RGB Image", WINDOW_AUTOSIZE);
    imshow("Original RGB Image", imageRGB);

    // 2. Побудуйте зображення гістограми у вигляді графіка
    Histogram1D h;
    const Mat bwOriginalHistogram = h.getHistogram(imageBW);
    for (int i = 0; i < 256; i++) {
        cout << "Value " << i << " = " << bwOriginalHistogram.at<float>(i) << endl;
    }
//    namedWindow("BW Histogram");
//    imshow("BW Histogram", Histogram1D::getImageOfHistogram(bwOriginalHistogram, 1));

    // 3. Підібравши порог, отримайте двійкове зображення
    // з сегментацією фону та передньої частини
    Mat binarySegmentedBackBwImg; // Вихідний бінарний образ
    threshold(imageBW,
              binarySegmentedBackBwImg,
              70, // Порогове значення
              255,// Присвоєне значення пікселів
              THRESH_BINARY // Тип порогу
    );
//    namedWindow("Binary BW Image: Back Segm");
//    imshow("Binary BW Image: Back Segm", binarySegmentedBackBwImg);

    // 4. Використовуючи таблицю пошуку, зробіть негатив власного
    // тестового зображення та побудуйте його гістограму.
    Mat result;
    Histogram1D negative;

    Mat lutNegative(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        lutNegative.at<uchar>(i) = 255 - i;
    }

    LUT(imageRGB, lutNegative, result);

//    imshow("Negative Histogram", negative.getHistogramImage(result));
//    imshow("Negative BW Image", result);

    // 5. Підвищить контраст власного тестового зображення за допомогою
    // розтягування гістограми з відсотковим відсіченням 1...5%. Побудуйте
    // гістограму отриманого зображення
    Histogram1D imageBwColorReduceHisto;
    Mat imageBwColorReduce = imageBW.clone();
    colorReduce(imageBwColorReduce);

//    namedWindow("BW Image 1-5%", WINDOW_AUTOSIZE);
//    imshow("BW Image 1-5%", imageBwColorReduce);
//
//    imshow("BW Image 1-5% Histogram", imageBwColorReduceHisto.getHistogramImage(imageBwColorReduce));

    // 6. Вирівняйте гістограму власного тестового зображення за допомогою
    // власної функції OpenCV. Відобразіть результат та його гістограму.
    Histogram1D equalizeHisto;

    equalizeHist(imageBW, result);

//    namedWindow("Equalized BW Image", WINDOW_AUTOSIZE);
//    imshow("Equalized BW Image", imageBW);
//    imshow("Equalized BW Image Histogram", equalizeHisto.getHistogramImage(result));

    // 7. За допомогою власної функції OpenCV для зворотного проектування гістограми
    // виявіть певний вміст зображення. Побудуйте карту ймовірності та відобразіть
    // її у негативному вигляді: від яскравого (низька ймовірність приналежності
    // до опорної площі) до темного (висока ймовірність).
    const Rect roi = Rect(0, 22, 160, 160);
    Mat imageROI = imageBW(roi);
    LUT(imageROI, lutNegative, result);

    Mat hist = negative.getHistogram(result);
    normalize(hist, hist, 1.0);

    int channels[] = {0};
    float h_range[] = {0, 255};
    const float *ranges[] = {h_range};

    Mat backProjImage;
    calcBackProject(&result, 1, channels, hist, backProjImage, ranges, 255.0, true);

//    namedWindow("BW Image ROI", WINDOW_AUTOSIZE);
//    imshow("BW Image ROI", imageROI);
//
//    namedWindow("Backproject BW Image", WINDOW_AUTOSIZE);
//    imshow("Backproject BW Image", backProjImage);

    // 8. Застосуйте поріг до попереднього зображення та отримайте зображення
    // з найбільш ймовірними пікселями опорної площі.
    Mat thresBackProjImg;
    cv::threshold(backProjImage, thresBackProjImg, 70, 255, cv::THRESH_BINARY);

//    namedWindow("Threshold Bp BW Image", WINDOW_AUTOSIZE);
//    imshow("Threshold Bp BW Image", thresBackProjImg);

    // 9. За допомогою власної функції OpenCV для зворотного проектування
    // гістограми виявіть певний вміст кольорової версії зображення (наприклад,
    // область блакитного неба).

    ColorHistogram hc;
    hc.setSize(8); // (8 бiнiв на канал), 8x8x8

    imageROI = imageRGB(cv::Rect(0, 22, 160, 160));
    cv::Mat rgbHisto = hc.getHistogram(imageROI);

    ContentFinder finder;
    finder.setHistogram(rgbHisto);
    finder.setThreshold(0.05f);
    result = finder.find(imageRGB);

//    namedWindow("Find RGB Image", WINDOW_AUTOSIZE);
//    imshow("Find RGB Image", result);

    // 10. Повторіть попередню вправу за допомогою обчислення розрідженої гістограми.

    Mat eqImageROI;
    equalizeHist(imageROI, eqImageROI);
    Mat eqRgbHisto = hc.getHistogram(eqImageROI);

    finder.setHistogram(eqRgbHisto);
    result = finder.find(imageRGB);

    namedWindow("Find Eq RGB Image ROI", WINDOW_AUTOSIZE);
    imshow("Find RGB Eq Image ROI", result);

    // 11. Проведіть експерименти, вирішуючи попередні задачі із застосуванням
    // колірного простору HSV, L*a*b* або ін.


    // 12. Застосовуючи алгоритмом середнього зсуву, з початкової прямокутної області
    // (тобто положення особи людини на початковому зображенні), відтворіть прямий
    // об'єкт на місці обличчя нової людини на іншовому зображенні.

    // 13. Застосувати поріг до зображення книги.

    // 14. Отримати двійкове зображення книги, використовуючи адаптивний поріг.

    // 15. Визначити розташування певного обєкту на зображенні, відстежуючи
    // його за допомогою гістограм.

    waitKey(0);
    return 0;
}