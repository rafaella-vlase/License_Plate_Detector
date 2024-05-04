#include "stdafx.h"
#include "common.h"
#include "fstream"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>
#include <tesseract/baseapi.h>
#include <cmath>
#include <leptonica/allheaders.h>
#include <regex>

using namespace cv;
using namespace std;

float calculScalar(int Kernel[3][3]) {
    int sum1 = 0;
    int sum2 = 0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (Kernel[i][j] < 0) {
                sum1 += abs(Kernel[i][j]);
            }
            else {
                sum2 += Kernel[i][j];
            }
        }
    }

    float scalar = max(sum1, sum2);
    return scalar;
}

Mat convolution(Mat src, int kernel[][3], int scalar)
{
    Mat dst = src.clone();

    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            int sum = 0;
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    int pixel = src.at<uchar>(i + k - 1, j + l - 1);
                    int filter = kernel[k][l] * pixel;
                    sum += filter;
                }
            }

            dst.at<uchar>(i, j) = sum / scalar;
        }
    }

    return dst;
}

Mat filtruGaussian(Mat src) {
    int Kernel[3][3] = { {1, 2, 1},
                        {2, 4, 2},
                        {1, 2, 1} };
    float scalar = calculScalar(Kernel);

    Mat dst = convolution(src, Kernel, scalar);

    return dst;
}

Mat connectEdges(Mat src) {
    int height = src.rows;
    int width = src.cols;

    int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    Mat dst = src.clone();

    for (int i = 1; i < height - 1; i++) {
        std::cout << "Connecting edges " << (i * 100) / height << "%\n";
        for (int j = 1; j < width - 1; j++) {
            if (dst.at<float>(i, j) == 255) {
                queue<pair<int, int>> q;
                q.push(make_pair(i, j));
                while (!q.empty()) {
                    pair<int, int> punct = q.front();
                    q.pop();
                    for (int k = 0; k < 8; k++) {
                        if (punct.first + di[k] < height && punct.second + dj[k] < width) {
                            if (dst.at<float>(punct.first + di[k], punct.second + dj[k]) == 128) {
                                dst.at<float>(punct.first + di[k], punct.second + dj[k]) = 255;
                                q.push(make_pair(punct.first + di[k], punct.second + dj[k]));
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (dst.at<float>(i, j) == 128) {
                dst.at<float>(i, j) = 0;
            }
        }
    }
    return dst;
}



cv::Mat canny(cv::Mat src, double weak_th = -1, double strong_th = -1) {
    src = filtruGaussian(src.clone());
    std::cout << "Gaussian filter - finished" << std::endl;
    //compute gradients
    cv::Mat gx, gy;
    cv::Sobel(src, gx, CV_64F, 1, 0, 3);
    cv::Sobel(src, gy, CV_64F, 0, 1, 3);

    cv::Mat mag, ang;
    cv::cartToPolar(gx, gy, mag, ang, true);

    // setting the minimum and maximum thresholds for double thresholding
    double minVal, maxVal;
    cv::minMaxLoc(mag, &minVal, &maxVal);
    double mag_max = maxVal;
    if (weak_th == -1) weak_th = mag_max * 0.1;
    if (strong_th == -1) strong_th = mag_max * 0.5;

    for (int i_x = 0; i_x < src.cols; ++i_x) {
        for (int i_y = 0; i_y < src.rows; ++i_y) {

            double grad_ang = ang.at<double>(i_y, i_x);
            if (abs(grad_ang) > 180)
                grad_ang = abs(grad_ang - 180);
            else 
                grad_ang = abs(grad_ang);

            // selecting the neighbours of the target pixel
            // according to the gradient direction
            // In the x axis direction
            int neighb_1_x, neighb_1_y, neighb_2_x, neighb_2_y;
            if (grad_ang <= 22.5) {
                neighb_1_x = i_x - 1; neighb_1_y = i_y;
                neighb_2_x = i_x + 1; neighb_2_y = i_y;
            }
            // top right (diagonal-1) direction
            else if (grad_ang > 22.5 && grad_ang <= (22.5 + 45)) {
                neighb_1_x = i_x - 1; neighb_1_y = i_y - 1;
                neighb_2_x = i_x + 1; neighb_2_y = i_y + 1;
            }
            // In y-axis direction
            else if (grad_ang > (22.5 + 45) && grad_ang <= (22.5 + 90)) {
                neighb_1_x = i_x; neighb_1_y = i_y - 1;
                neighb_2_x = i_x; neighb_2_y = i_y + 1;
            }
            // top left (diagonal-2) direction
            else if (grad_ang > (22.5 + 90) && grad_ang <= (22.5 + 135)) {
                neighb_1_x = i_x - 1; neighb_1_y = i_y + 1;
                neighb_2_x = i_x + 1; neighb_2_y = i_y - 1;
            }
            // Now it restarts the cycle
            else if (grad_ang > (22.5 + 135) && grad_ang <= (22.5 + 180)) {
                neighb_1_x = i_x - 1; neighb_1_y = i_y;
                neighb_2_x = i_x + 1; neighb_2_y = i_y;
            }

            // Non-maximum suppression step
            if (src.cols > neighb_1_x >= 0 && src.rows > neighb_1_y >= 0) {
                if (mag.at<double>(i_y, i_x) < mag.at<double>(neighb_1_y, neighb_1_x)) {
                    mag.at<double>(i_y, i_x) = 0;
                    continue;
                }
            }

            if (src.cols > neighb_2_x >= 0 && src.rows > neighb_2_y >= 0) {
                if (mag.at<double>(i_y, i_x) < mag.at<double>(neighb_2_y, neighb_2_x)) {
                    mag.at<double>(i_y, i_x) = 0;
                }
            }
        }
    }

    std::cout << "Non-maximum suppresion - finished" << std::endl;

    // double thrsholding step
    for (int i_y = 0; i_y < mag.rows; ++i_y) {
        for (int i_x = 0; i_x < mag.cols; ++i_x) {

            if (mag.at<double>(i_y, i_x) < weak_th) {
                mag.at<double>(i_y, i_x) = 0;
            }
            else if (strong_th > mag.at<double>(i_y, i_x) >= weak_th) {
                mag.at<double>(i_y, i_x) = 128;
            }
            else {
                mag.at<double>(i_y, i_x) = 255;
            }
        }
    }

    std::cout << "Double-tresholding - finished" << std::endl;
    // finally returning the magnitude of
    // gradients of edges
    mag = connectEdges(mag.clone());
    std::cout << "Connecting edges - finished" << std::endl;
    return mag;
}

bool compareContourAreas(const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
    return cv::contourArea(c1, false) > cv::contourArea(c2, false);
}

int main() {
    Mat img = imread("./images/qashqai.jpg");
    if (img.empty()) {
        std::cerr << "Error opening image file\n";
        return -1;
    }

    Mat thresh;
    if (img.channels() == 3) {
        cvtColor(img, thresh, COLOR_BGR2GRAY);
    }
    else {
        thresh = img.clone();
    }

    adaptiveThreshold(thresh, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

    Mat edged;
    Canny(thresh, edged, 30, 200);
    std::cout << "Edge detection - finished\n";

    vector<vector<Point>> contours;
    findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Rect license_plate_rect;
    for (const vector<Point>& contour : contours) {
        double area = contourArea(contour);
        double aspect_ratio = float(boundingRect(contour).width) / boundingRect(contour).height;
        vector<Point> hull;
        convexHull(contour, hull);
        double hull_area = contourArea(hull);
        double solidity = area / hull_area;

        if (area > 100 && aspect_ratio > 2 && aspect_ratio < 6 && solidity > 0.7) {
            license_plate_rect = boundingRect(contour);
            break;
        }
    }

    Mat license_plate_image;
    if (!license_plate_rect.empty()) {
        license_plate_image = img(license_plate_rect);
    }

    if (!license_plate_image.empty()) {
        tesseract::TessBaseAPI* tess = new tesseract::TessBaseAPI();
        tess->Init("C:/Users/ella/vcpkg/packages/tesseract_x64-windows-static/share/tessdata", "eng", tesseract::OEM_DEFAULT);
        tess->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

        int width = license_plate_image.cols;
        int height = license_plate_image.rows;
        uchar* license_plate_data = license_plate_image.data;

        tess->SetImage(license_plate_data, width, height, license_plate_image.channels(), width);
        char* ocr_result = tess->GetUTF8Text();

        std::cout << "License plate number: " << ocr_result << std::endl;

        tess->End();
        delete tess;
        free(ocr_result);
    }
    else {
        std::cout << "License plate not found in the image.\n";
    }

    // Display the original image, edged image, and cropped license plate (if found)
    imshow("Original Image", img);
    if (!edged.empty()) {
        imshow("Edged Image", edged);
    }
    if (!license_plate_image.empty()) {
        imshow("License Plate", license_plate_image);
    }
    waitKey(0);
    return 0;
}