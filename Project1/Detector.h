#pragma once
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <string>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class RoadLaneDetector
{

public:
	Mat filter_colors(Mat img_frame);
	Mat limit_region(Mat img_edges);
	vector<Vec4i> houghLines(Mat img_mask);
	vector<vector<Vec4i> > separateLine(Mat img_edges, vector<Vec4i> lines);
	vector<Point> regression(vector<vector<Vec4i> > separated_lines, Mat img_input);
	string predictDir();
	Mat drawLine(Mat img_input, vector<Point> lane, string dir);

private:
	double size, center;
	double left_m, right_m;
	Point left_b, right_b;
	bool left_detect = false, right_detect = false;

	//���� ���� ���� ���� ��� 
	double poly_bottom_width = 0.83;  //��ٸ��� �Ʒ��� �����ڸ� �ʺ� ����� ���� �����
	double poly_top_width = 0.06;     //��ٸ��� ���� �����ڸ� �ʺ� ����� ���� �����
	double poly_height = 0.3;         //��ٸ��� ���� ����� ���� �����


};
#pragma once
