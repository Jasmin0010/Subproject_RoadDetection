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
	string dir;

	VideoCapture video("clip2.mp4");  //���� �ҷ�����

	if (!video.isOpened())
	{
		cout << "������ ������ �� �� �����ϴ�. \n" << endl;
		getchar();
		return -1;
	}

	video.read(img_frame);
	if (img_frame.empty()) return -1;


	VideoWriter writer;
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  //���ϴ� �ڵ� ����
	double fps = video.get(CAP_PROP_FPS);  //������
	string filename = "./result.avi";  //��� ����
	int delay = cvRound(1000 / fps);
	

	writer.open(filename, codec, fps, img_frame.size(), CV_8UC3);
	if (!writer.isOpened()) {
		cout << "����� ���� ���� ������ �� �� �����ϴ�. \n";
		return -1;
	}

	video.read(img_frame);
	int cnt = 0;

	while (1) {
		//1. ���� ������ �о�´�.
		if (!video.read(img_frame)) break;

		result = img_frame.clone();

		Mat ycbcr, ycbcr_channel[3];
		cvtColor(img_frame, ycbcr, COLOR_BGR2YCrCb);

		split(ycbcr, ycbcr_channel);
		equalizeHist(ycbcr_channel[0], ycbcr_channel[0]);
		merge(ycbcr_channel, 3, ycbcr);

		cvtColor(ycbcr, img_frame, COLOR_YCrCb2BGR);


		//2. ���, ����� ���� ���� �ִ� �͸� ���͸��Ͽ� ���� �ĺ��� �����Ѵ�.
		img_filter = roadLaneDetector.filter_colors(img_frame);

		//3. ������ GrayScale ���� ��ȯ�Ѵ�.
		cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);

		//4. Canny Edge Detection���� ������ ����. (���� ���Ÿ� ���� Gaussian ���͸��� ����)
		Canny(img_filter, img_edges, 50, 150);

		//5. �ڵ����� ������� �ٴڿ� �����ϴ� �������� �����ϱ� ���� ���� ������ ����
		img_mask = roadLaneDetector.limit_region(img_edges);

		//6. Hough ��ȯ���� ���������� ���� ������ ����
		lines = roadLaneDetector.houghLines(img_mask);

		if (lines.size() > 0) {
			//7. ������ ������������ �¿� ������ ���� ���ɼ��� �ִ� �����鸸 ���� �̾Ƽ� �¿� ���� ������ ����Ѵ�. 
			// ���� ȸ�͸� �Ͽ� ���� ������ ���� ã�´�.
			separated_lines = roadLaneDetector.separateLine(img_mask, lines);
			lane = roadLaneDetector.regression(separated_lines, img_frame);

			//8. ���� ���� ����
			dir = roadLaneDetector.predictDir();

			//9. ���� ���� ������ ������ �׸��� ���� �ٰ����� ������ ä���. ���� ���� ���� �ؽ�Ʈ�� ���� ���
			image = roadLaneDetector.drawLine(result, lane, dir);
			//image = roadLaneDetector.drawLine(image, lane, dir);
		}

		//10. ����� ������ ���Ϸ� ����. ĸ���Ͽ� ���� ����
		//writer << image;
		//if (cnt++ == 15)
		//	imwrite("image_result.jpg", image);  //ĸ���Ͽ� ���� ����

		//11. ��� ���� ���
		//resize(image, image, Size(1024, 720), 0, 0, INTER_LINEAR);
		resize(result, result, Size(1024, 720), 0, 0, INTER_LINEAR);

		//imshow("image_eq", image);
		imshow("result", result);

		//esc Ű ����
		if (waitKey(delay) == 27)
			break;
	}
	return 0;
}

//Mat img_frame, result;

//int main(int argc, char** argv) {
//
//	char* imageName = argv[1];
//	img_frame = imread(imageName, IMREAD_COLOR);
//
//	if (argc != 2 || !img_frame.data) {
//		printf(" No image data \n ");
//		return -1;
//	}
//
//	result = img_frame.clone();
//
//	waitKey(0);
//
//	/*namedWindow("origin");
//	imshow("origin", img_frame);*/
//
//	RoadLaneDetector roadLaneDetector;
//	Mat img_filter, img_edges, img_mask, img_lines, image;
//	vector<Vec4i> lines;
//	vector<vector<Vec4i> > separated_lines;
//	vector<Point> lane;
//	string dir;
//
//	//������׷� ��Ȱȭ �ϱ�
//	Mat ycbcr, ycbcr_channel[3];
//	cvtColor(img_frame, ycbcr, COLOR_BGR2YCrCb);
//
//	split(ycbcr, ycbcr_channel);
//	equalizeHist(ycbcr_channel[0], ycbcr_channel[0]);
//	merge(ycbcr_channel, 3, ycbcr);
//
//	cvtColor(ycbcr, img_frame, COLOR_YCrCb2BGR);
//
//
//	//2. ���, ����� ���� ���� �ִ� �͸� ���͸��Ͽ� ���� �ĺ��� �����Ѵ�.
//	img_filter = roadLaneDetector.filter_colors(img_frame);
//
//	//3. ������ GrayScale ���� ��ȯ�Ѵ�.
//	cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);
//
//	//4. Canny Edge Detection���� ������ ����. (���� ���Ÿ� ���� Gaussian ���͸��� ����)
//	Canny(img_filter, img_edges, 50, 150);
//
//	//5. �ڵ����� ������� �ٴڿ� �����ϴ� �������� �����ϱ� ���� ���� ������ ����
//	img_mask = roadLaneDetector.limit_region(img_edges);
//
//	//6. Hough ��ȯ���� ���������� ���� ������ ����
//	lines = roadLaneDetector.houghLines(img_mask);
//
//	if (lines.size() > 0) {
//		//7. ������ ������������ �¿� ������ ���� ���ɼ��� �ִ� �����鸸 ���� �̾Ƽ� �¿� ���� ������ ����Ѵ�. 
//		// ���� ȸ�͸� �Ͽ� ���� ������ ���� ã�´�.
//		separated_lines = roadLaneDetector.separateLine(img_mask, lines);
//		lane = roadLaneDetector.regression(separated_lines, img_frame);
//
//		//8. ���� ���� ����
//		dir = roadLaneDetector.predictDir();
//
//		//9. ���� ���� ������ ������ �׸��� ���� �ٰ����� ������ ä���. ���� ���� ���� �ؽ�Ʈ�� ���� ���
//		image = roadLaneDetector.drawLine(result, lane, dir);
//		//image = roadLaneDetector.drawLine(image, lane, dir);
//	}
//
//	//resize(image, image, Size(1024, 720), 0, 0, INTER_LINEAR);
//	resize(result, result, Size(1024, 720), 0, 0, INTER_LINEAR);
//
//	//imshow("image_eq", image);
//	imshow("result", result);
//
//	waitKey(0);
//
//	destroyAllWindows();
//
//	return 0;
//}