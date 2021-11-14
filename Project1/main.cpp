#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Detector.h"
#include <opencv2/imgproc/imgproc.hpp>  


int main()
{
	RoadLaneDetector roadLaneDetector;
	Mat img_frame, img_filter, img_edges, img_mask, img_lines, image, result;
	vector<Vec4i> lines;
	vector<vector<Vec4i> > separated_lines;
	vector<Point> lane;
	bool isitstraight;

	VideoCapture video("clip2.mp4");  

	if (!video.isOpened())
	{
		cout << "동영상 파일을 열 수 없습니다. \n" << endl;
		getchar();
		return -1;
	}

	video.read(img_frame);
	if (img_frame.empty()) return -1;


	VideoWriter writer;
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  //원하는 코덱 선택
	double fps = video.get(CAP_PROP_FPS);  //프레임
	string filename = "./clip1_result.avi";  //결과 파일
	int delay = cvRound(1000 / fps);
	

	writer.open(filename, codec, fps, img_frame.size(), CV_8UC3);
	if (!writer.isOpened()) {
		cout << "출력을 위한 비디오 파일을 열 수 없습니다. \n";
		return -1;
	}

	video.read(img_frame);
	int cnt = 0;

	while (1) {
		if (!video.read(img_frame)) break;

		result = img_frame.clone();

		Mat ycbcr, ycbcr_channel[3];
		cvtColor(img_frame, ycbcr, COLOR_BGR2YCrCb);

		split(ycbcr, ycbcr_channel);
		equalizeHist(ycbcr_channel[0], ycbcr_channel[0]);
		merge(ycbcr_channel, 3, ycbcr);

		cvtColor(ycbcr, img_frame, COLOR_YCrCb2BGR);

		img_filter = roadLaneDetector.filter_colors(img_frame);

		cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);

		Canny(img_filter, img_edges, 50, 150);

		img_mask = roadLaneDetector.limit_region(img_edges);

		lines = roadLaneDetector.houghLines(img_mask);

		if (lines.size() > 0) {
			
			separated_lines = roadLaneDetector.separateLine(img_mask, lines);
			lane = roadLaneDetector.regression(separated_lines, img_frame);

			isitstraight = roadLaneDetector.predictDir();

			image = roadLaneDetector.drawLine(result, lane, isitstraight);
		}

		writer << image;

		resize(result, result, Size(1024, 720), 0, 0, INTER_LINEAR);

		imshow("result", result);

		if (waitKey(delay) == 27)
			break;
	}
	return 0;
}
