#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "Detector.h"
#include <iostream>
#include <string>
#include <vector>


Mat RoadLaneDetector::filter_colors(Mat img_frame) {
	
	Mat output, print;
	UMat img_hsv;
	UMat white_mask, white_image;
	UMat yellow_mask, yellow_image;
	UMat blue_mask, blue_image;
	img_frame.copyTo(output);

	//차선 색깔 범위 
	Scalar lower_white = Scalar(150, 150, 150); //white range (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //yellow range(HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);
	Scalar lower_blue = Scalar(80, 40, 50);		//blue range(HSV) 
	Scalar upper_blue = Scalar(130, 100, 150);

	//white
	inRange(output, lower_white, upper_white, white_mask);
	bitwise_and(output, output, white_image, white_mask);

	cvtColor(output, img_hsv, COLOR_BGR2HSV);

	//yellow
	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and(output, output, yellow_image, yellow_mask);

	//blue
	inRange(img_hsv, lower_blue, upper_blue, blue_mask);
	bitwise_and(output, output, blue_image, blue_mask);

	//merge
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, output);
	addWeighted(output, 1.0, blue_image, 1.0, 0.0, output);

	output.copyTo(print);
	resize(print, print, Size(1024, 720), 0, 0, INTER_LINEAR);
	namedWindow("color_selected image");
	imshow("color_selected image", print);
	return output;
}


Mat RoadLaneDetector::limit_region(Mat img_edges) {

	int width = img_edges.cols;
	int height = img_edges.rows;

	Mat output;
	Mat mask = Mat::zeros(height, width, CV_8UC1);

	//masking interesting part
	Point points[4]{
		Point((width * (1 - poly_bottom_width)) / 2, height),
		Point((width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_bottom_width)) / 2, height)
	};

	fillConvexPoly(mask, points, 4, Scalar(255, 0, 0));

	bitwise_and(img_edges, mask, output);
	return output;
}

vector<Vec4i> RoadLaneDetector::houghLines(Mat img_mask) {
	/*
		관심영역으로 마스킹 된 이미지에서 모든 선을 추출하여 반환
	*/
	vector<Vec4i> line;

	//확률적용 허프변환 직선 검출 함수 
	HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> RoadLaneDetector::separateLine(Mat img_edges, vector<Vec4i> lines) {

	vector<vector<Vec4i>> output(2);
	Point p1, p2;
	vector<double> slopes;
	vector<Vec4i> final_lines, left_lines, right_lines;
	double slope_thresh = 0.3;

	//find gradient
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		p1 = Point(line[0], line[1]);
		p2 = Point(line[2], line[3]);

		double slope;
		if (p2.x - p1.x == 0)  //if it is corner
			slope = 999.0;
		else
			slope = (p2.y - p1.y) / (double)(p2.x - p1.x);

		if (abs(slope) > slope_thresh) {//horizon part
			slopes.push_back(slope);
			final_lines.push_back(line);
		}
	}

	center = (double)((img_edges.cols / 2));

	for (int i = 0; i < final_lines.size(); i++) {
		p1 = Point(final_lines[i][0], final_lines[i][1]);
		p2 = Point(final_lines[i][2], final_lines[i][3]);

		if (slopes[i] > 0 && p1.x > center && p2.x > center) {
			right_detect = true;
			right_lines.push_back(final_lines[i]);
		}
		else if (slopes[i] < 0 && p1.x < center && p2.x < center) {
			left_detect = true;
			left_lines.push_back(final_lines[i]);
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}

vector<Point> RoadLaneDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
	
	vector<Point> output(4);
	Point p1, p2, p3, p4;
	Vec4d left_line, right_line;
	vector<Point> left_points, right_points;

	if (right_detect) {
		for (auto i : separatedLines[0]) {
			p1 = Point(i[0], i[1]);
			p2 = Point(i[2], i[3]);

			right_points.push_back(p1);
			right_points.push_back(p2);
		}

		if (right_points.size() > 0) {

			fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);

			right_m = right_line[1] / right_line[0];  //gradient
			right_b = Point(right_line[2], right_line[3]);
		}
	}

	if (left_detect) {
		for (auto j : separatedLines[1]) {
			p3 = Point(j[0], j[1]);
			p4 = Point(j[2], j[3]);

			left_points.push_back(p3);
			left_points.push_back(p4);
		}

		if (left_points.size() > 0) {
			
			fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //gradient
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	//y = m*x + b  --> x = (y-b) / m
	int y1 = img_input.rows; //밑의 y좌표
	int y2 = 470; //위의 y좌표

	double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
	double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

	double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
	double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_x1, y1);
	output[1] = Point(right_x2, y2);//위
	output[2] = Point(left_x1, y1);
	output[3] = Point(left_x2, y2);//위

	return output;
}

bool RoadLaneDetector::predictDir() {

	string output;
	double x, right_threshold = 100, left_threshold = 120;

	//두 차선이 교차하는 지점 계산
	x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	if (x >= (center - left_threshold) && x <= (center + right_threshold)) {
		return true;
	}
	else {
		return false;
	}


}

Mat RoadLaneDetector::drawLine(Mat img_input, vector<Point> lane, bool isitstraight) {

	vector<Point> poly_points;
	Mat output;
	img_input.copyTo(output);

	double y_threshold = img_input.rows * 3 / 4;

	if (lane[1].y < y_threshold) {
		double right_line_gradient = (double)(lane[0].y - lane[1].y) / (double)(lane[0].x - lane[1].x);

		double fixed_lane1_x = (double)(y_threshold - lane[1].y) / right_line_gradient + lane[1].x;
		lane[1].x = fixed_lane1_x;
		lane[1].y = y_threshold;
	}

	if (lane[3].y < y_threshold) {
		double left_line_gradient = (double)(lane[2].y - lane[3].y) / (double)(lane[3].x - lane[2].x);

		double fixed_lane3_x = (double)(lane[3].y - y_threshold) / left_line_gradient + lane[3].x;
		lane[3].x =fixed_lane3_x;
		lane[3].y = y_threshold;
	}


	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);

	fillConvexPoly(output, poly_points, Scalar(230, 30, 0), LINE_AA, 0);  
	addWeighted(output, 0.3, img_input, 0.7, 0, img_input);  //merge

	//예측 진행 방향
	if (isitstraight) {
		putText(img_input, "Good", Point(520, 100), FONT_HERSHEY_PLAIN, 3, Scalar(255, 255, 255), 3, LINE_AA);

		//좌우 차선 선 그리기
		line(img_input, lane[0], lane[1], Scalar(0, 255, 255), 5, LINE_AA);
		line(img_input, lane[2], lane[3], Scalar(0, 255, 255), 5, LINE_AA);
	}
	else {
		putText(img_input, "Warning", Point(520, 100), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 3.8, LINE_AA);
		
		//좌우 차선 선 그리기
		line(img_input, lane[0], lane[1], Scalar(0, 0, 255), 5, LINE_AA);
		line(img_input, lane[2], lane[3], Scalar(0, 0, 255), 5, LINE_AA);
	}
	

	return img_input;

}