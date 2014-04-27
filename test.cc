#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

const int WIDTH  = 800;
const int HEIGHT = 600;

pair<double, double> mean_std(const vector<double> & data)
{
  double sum = accumulate(data.begin(), data.end(), 0.0);
  double mean = sum / data.size();

  double accum = 0.0;
  for_each(
      data.begin(),
      data.end(), 
      [&](const double d)
      {
        accum += (d - mean) * (d - mean);
      }
    );

  double stddev = sqrt(accum/(data.size()-1));
  return make_pair(mean, stddev);
}

int main(int argc, char **argv)
{
  VideoCapture capture(1);
  capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, 600);

  namedWindow("Optical flow", CV_WINDOW_AUTOSIZE);
  namedWindow("Parameters", CV_WINDOW_AUTOSIZE);

  int qualityLevel = 1;
  int minDistance  = 1;
  int maxFeatures  = 500;
  createTrackbar("Quality", "Parameters", &qualityLevel, 1000);
  createTrackbar("Minimum Distance", "Parameters", &minDistance, 1000);
  createTrackbar("Number of features", "Parameters", &maxFeatures, 500);

  Mat prevFrame, prevGray;
  capture >> prevFrame;
  cvtColor(prevFrame, prevGray, CV_BGR2GRAY);

  vector<Point2f> corners;
  vector<Point2f> nextCorners;
  vector<unsigned char> status;
  vector<float> err;

  int frameCount = 0;
  auto beginning = chrono::system_clock::now();
  while (true)
  {
    if (cvWaitKey(33) == 27)
      break;

    auto start = chrono::system_clock::now();

    Mat curFrame, curGray;
    capture >> curFrame;
    cvtColor(curFrame, curGray, CV_BGR2GRAY);

    if (qualityLevel == 0) qualityLevel = 1;
    if (minDistance == 0) minDistance = 1;

    if (frameCount == 0 || frameCount++ == 5)
    {
      auto now = chrono::system_clock::now();
      //cout << "Making new features @ " << chrono::duration_cast<chrono::seconds>(now-beginning).count() << endl;

      goodFeaturesToTrack(curGray, corners, float(maxFeatures), float(qualityLevel)/1000, float(minDistance)/1000);
      cornerSubPix(curGray, corners, Size(15, 15), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
      frameCount = 1;
    }

    calcOpticalFlowPyrLK(prevGray, curGray, corners, nextCorners, status, err);
    //int trackedFeatures = accumulate(status.begin(), status.end(), 0);
    //cout << trackedFeatures << " features successfully tracked" << endl;

    vector<double> angles;
    vector<double> lengths;
    for (int i = 0; i < nextCorners.size(); i++)
    {
      if (status[i] == 0)
        continue;

      Point2f p = corners[i];
      Point2f q = nextCorners[i];

      double angle = atan2(q.y - p.y, q.x - p.x) * 180/CV_PI;
      double length = sqrt(pow(p.x-q.x, 2) + pow(p.y-q.y, 2));

      angles.push_back(angle);
      lengths.push_back(length);
    }
    auto angle_stats  = mean_std(angles);
    auto length_stats = mean_std(lengths);

    cout << "Angle  (mean, std): (" << angle_stats.first << ", " << angle_stats.second << ")" << endl;
    cout << "Length (mean, std): (" << length_stats.first << ", " << length_stats.second << ")" << endl;

    int discarded = 0;
    for (int i = 0; i < nextCorners.size(); i++)
    {
      if (status[i] == 0)
        continue;

      Point2f p = corners[i];
      Point2f q = nextCorners[i];

      double angle = atan2(q.y - p.y, q.x - p.x) * 180/CV_PI;
      double length = sqrt(pow(p.x-q.x, 2) + pow(p.y-q.y, 2));

      if (angle > angle_stats.first - angle_stats.second || angle < angle_stats.first + angle_stats.second && 
          length > length_stats.first - length_stats.second || length < length_stats.first + length_stats.second)
        line(curFrame, p, q, CV_RGB(255, 0, 0), 1);
      else
        discarded++;
    }

    cout << "Discarded " << discarded << " outliers." << endl;

    imshow("Optical flow", curFrame);
    prevGray = curGray;

    auto end = chrono::system_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);

    //cout << elapsed.count() << "ms" << endl;
  }
}
