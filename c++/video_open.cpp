#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
	cv::namedWindow("Frame",cv::WINDOW_AUTOSIZE);
	cv::VideoCapture cap;
	cap.open(std::string (argv[1]));
	cv::Mat frame;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		cv::imshow("Video_Frame", frame);
		if (cv::waitKey(10) >= 0)
		{
			break;
		}
	}
	return 0;
}