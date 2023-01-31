#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
	std::string path = "test.jpg";
	cv::Mat img = cv::imread(path);
	if (img.empty())
	{
		return -1;
	}
	cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
	cv::imshow("Frame", img);
	cv::waitKey(0);
	return 0;
}