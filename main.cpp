#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const cv::String picPath = R"(pics\hot-charcoal.jpg)";
const cv::String bookPath = R"(pics\book.png)";
const cv::String img1Path = R"(pics\1.png)";
const cv::String img2Path = R"(pics\2.png)";

void integralThreshold(const Mat iimage, Mat binary, int blockSize = 21, int threshold = 10) {
    int halfSize = blockSize / 2;
    int nl = iimage.rows;
    int nc = iimage.cols * iimage.channels();

    for (int j = halfSize; j < nl - halfSize - 1; j++) {

        auto *data = binary.ptr<uchar>(j);
        const int *idata1 = iimage.ptr<int>(j - halfSize);
        const int *idata2 = iimage.ptr<int>(j + halfSize + 1);

        // Для кожного пікселя рядка
        for (int i = halfSize; i < nc - halfSize - 1; i++) {

            // Обчислювальна сума
            int sum = (idata2[i + halfSize + 1] - idata2[i - halfSize] -
                       idata1[i + halfSize + 1] + idata1[i - halfSize])
                      / (blockSize * blockSize);

            // адаптивний поріг
            if (data[i] < (sum - threshold))
                data[i] = 0;
            else
                data[i] = 255;
        }
    }
}

class ContentFinder {
private:
    float hranges[2];
    const float *ranges[3];
    int channels[3];
    float threshold;
    cv::Mat histogram;
    cv::SparseMat shistogram;
    bool sparse;
public:
    ContentFinder() : threshold(0.1f), sparse(false) {
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
    }

    void setHistogram(const cv::Mat &h) {
        sparse = false;
        histogram = h;
        cv::normalize(histogram, histogram, 1.0);
    }

    void setHistogram(const cv::SparseMat &h) {
        sparse = true;
        shistogram = h;
        cv::normalize(shistogram, shistogram, 1.0, NORM_L2);
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

        if (sparse) {
            cv::calcBackProject(&image, 1, channels, shistogram, result, ranges, 255.0);
        } else {
            cv::calcBackProject(&image, 1, channels, histogram, result, ranges, 255.0);
        }

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
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;

        cv::Mat hist;
        cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
        return hist;
    }

    cv::SparseMat getSparseHistogram(const cv::Mat &image) {
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;

        cv::SparseMat hist(3, histSize, CV_32F);
        cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
        return hist;
    }

    cv::Mat getHueHistogram(const cv::Mat &image, int minSaturation = 0) {
        cv::Mat hist;

        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;

        if (minSaturation > 0) {
            // Spliting the 3 channels into 3 images
            std::vector<cv::Mat> v;
            cv::split(hsv, v);
            // Mask out the low saturated pixels
            cv::threshold(v[1], mask, minSaturation, 255, cv::THRESH_BINARY);
        }

        // Prepare arguments for a 1D hue histogram
        hranges[0] = 0.0;
        hranges[1] = 180.0;
        channels[0] = 0; // the hue channel

        // Compute histogram
        cv::calcHist(&hsv,
                     1,        // histogram of 1 image only
                     channels, // the channel used
                     mask,     // no mask is used
                     hist,     // the resulting histogram
                     1,        // it is a 1D histogram
                     histSize, // number of bins
                     ranges    // pixel value range
        );

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

    void setSize(int size) {
        histSize[0] = size;
    }
};

class ImageComparator {
private:
    cv::Mat refH;
    cv::Mat inputH;
    ColorHistogram hist;
    int nBins;
public:
    ImageComparator() : nBins(8) {
    }

    void setReferenceImage(const cv::Mat &image) {
        hist.setSize(nBins);
        refH = hist.getHistogram(image);
    }

    double compare(const cv::Mat &image) {
        inputH = hist.getHistogram(image);
        return cv::compareHist(refH, inputH, cv::HISTCMP_INTERSECT);
    }
};

template<typename T, int N>
class IntegralImage {
    cv::Mat integralImage;
public:
    IntegralImage(cv::Mat image) {
        // (costly) computation of the integral image
        cv::integral(image, integralImage, cv::DataType<T>::type);
    }

    // compute sum over sub-regions of any size from 4 pixel accesses
    cv::Vec<T, N> operator()(int xo, int yo, int width, int height) {
        // window at (xo,yo) of size width by height
        return (integralImage.at<cv::Vec<T, N>>(yo + height, xo + width)
                - integralImage.at<cv::Vec<T, N>>(yo + height, xo)
                - integralImage.at<cv::Vec<T, N>>(yo, xo + width)
                + integralImage.at<cv::Vec<T, N>>(yo, xo));
    }
};

void convertToBinaryPlanes(const cv::Mat &input, cv::Mat &output, int nPlanes) {
    int n = 8 - static_cast<int>( log(static_cast<double>(nPlanes)) / log(2.0));
    // mask used to eliminate least significant bits
    uchar mask = 0xFF << n;
    // create a vector of binary images
    std::vector<cv::Mat> planes;
    // reduce to nBins by eliminating least significant bits
    cv::Mat reduced = input & mask;

    // compute each binary image plane
    planes.reserve(nPlanes);
    for (int i = 0; i < nPlanes; i++) {
        // 1 for each pixel equals to i<<shift
        planes.push_back((reduced == (i << n)) & 0x1);
    }
    // create multi-channel image
    cv::merge(planes, output);
}

int main() {
    // 1. Завантажити зображення
    Mat imageBW = imread(picPath, 0);
//    namedWindow("Original BW Image", WINDOW_AUTOSIZE);
//    imshow("Original BW Image", imageBW);

    Mat imageRGB = imread(picPath);
//    namedWindow("Original RGB Image", WINDOW_AUTOSIZE);
//    imshow("Original RGB Image", imageRGB);

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
    const Rect2i roiBack = cv::Rect(0, 0, 80, 90);

    ColorHistogram hc;
    hc.setSize(8); // (8 бiнiв на канал), 8x8x8

    imageROI = imageRGB(roi);
    cv::Mat rgbHisto = hc.getHistogram(imageROI);

    ContentFinder finder;
    finder.setHistogram(rgbHisto);
    finder.setThreshold(0.05f);
    result = finder.find(imageRGB);

//    namedWindow("Find RGB Image", WINDOW_AUTOSIZE);
//    imshow("Find RGB Image", result);

    // 10. Повторіть попередню вправу за допомогою обчислення розрідженої гістограми.
    SparseMat spRgbHisto = hc.getSparseHistogram(imageROI);
    finder.setHistogram(spRgbHisto);
    finder.setThreshold(0.05f);
    result = finder.find(imageRGB);

//    namedWindow("Find Sp RGB Image ROI", WINDOW_AUTOSIZE);
//    imshow("Find Sp RGB Image ROI", result);

    // 11. Проведіть експерименти, вирішуючи попередні задачі із застосуванням
    // колірного простору HSV, L*a*b* або ін.

    Mat hsvRgbHisto = hc.getHueHistogram(imageROI);
    finder.setHistogram(hsvRgbHisto);
    finder.setThreshold(0.05f);
    result = finder.find(imageRGB);

//    namedWindow("Find HSV Image ROI", WINDOW_AUTOSIZE);
//    imshow("Find HSV Image ROI", result);

    // 12. Застосовуючи алгоритмом середнього зсуву, з початкової прямокутної області
    // (тобто положення особи людини на початковому зображенні), відтворіть прямий
    // об'єкт на місці обличчя нової людини на іншовому зображенні.
    Mat image1 = imread(img1Path);
    Rect image1RoiRect = Rect(100, 300, 40, 62);
    Mat image1roi = image1(image1RoiRect);

    cv::Mat image2 = cv::imread(img2Path);
    cv::Mat image2hsv;
    cv::cvtColor(image2, image2hsv, COLOR_BGR2HSV);

    cv::Mat img1RoiColorHist = hc.getHueHistogram(image1roi, 10);
    finder.setThreshold(0.0f);
    finder.setHistogram(img1RoiColorHist);

    int ch[1] = {0};
    result = finder.find(image2hsv, 0.0f, 180.0f, ch);

    // search objet with mean shift
    Rect trackWindow = Rect(image1RoiRect);
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER, 10000, 0.0001);
    cv::meanShift(result, trackWindow, criteria);

    cv::Mat imgResultSource = image1.clone();
    rectangle(imgResultSource, image1RoiRect, Scalar(255, 0), 1);
//    imshow("Source", imgResultSource);

    cv::Mat imgResultSearched = image2.clone();
    rectangle(imgResultSearched, trackWindow, Scalar(0, 0, 255), 2);
//    imshow("Found", imgResultSearched);

    // 13. Застосувати поріг до зображення книги.

    Mat imageBook = imread(bookPath);
    cv::cvtColor(imageBook, imageBook, COLOR_BGR2GRAY);
//    imshow("Original Book Image", imageBook);

    Mat binaryFixed;
    threshold(imageBook, binaryFixed, 70, 255, THRESH_BINARY);
//    imshow("Fixed Threshold Book Image", binaryFixed);

    // 14. Отримати двійкове зображення книги, використовуючи адаптивний поріг.

    Mat binaryAdaptive;
//    integralThreshold(imageBook, binaryAdaptive);
    cv::adaptiveThreshold(imageBook, binaryAdaptive, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                          21, 0);
//    namedWindow("Adaptive Threshold Book Image", WINDOW_AUTOSIZE);
//    imshow("Adaptive Threshold Book Image", binaryAdaptive);

    // 15. Визначити розташування певного обєкту на зображенні, відстежуючи
    // його за допомогою гістограм.

    Histogram1D h15;
    h15.setSize(16);

    const Rect area = image1RoiRect;

    // Обчислити гістограму зображення
    cv::Mat refHistogram= h15.getHistogram(image1roi);
    // Спочатку створити 16-плоский бінарний образ
    cv::Mat planes;
    convertToBinaryPlanes(image2,planes,16);
    // Потім обчислити цілісне зображення
    IntegralImage<float,16> intHistogram(planes);

    double maxSimilarity = 0.0;
    int xbest, ybest;
    // Контур над горизонтальною смугою навколо розташування людини на початковому зображенні
    for (int y = 110; y < 120; y++) {
        for (int x = 0; x < image2.cols - area.width; x++) {
            // Обчислити гістограму з 16 бiнів за допомогою цілісного зображення
            auto histogram = intHistogram(x, y, area.width, area.height);
            // Обчислити відстань від вихідної гістограми
            double distance = cv::compareHist(refHistogram,histogram, HISTCMP_INTERSECT);
            // Знаходимо положення найбільш подібної гістограми
            if (distance > maxSimilarity) {
                xbest = x;
                ybest = y;
                maxSimilarity = distance;
            }
        }
    }
    // намалюемо прямокутник у найкращому місці
    cv::rectangle(image2,cv::Rect(xbest, ybest, area.width, area.height), 0);
    imshow("Result", image2);

    waitKey(0);
    return 0;
}