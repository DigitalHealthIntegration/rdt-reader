#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "RdtProcessing.h"
using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {
	string window_name = "RDTInterface";
	VideoCapture stream1;
	if (argc > 1) {
		stream1.open(argv[1]);   //0 is the id of video device.0 if you have only one camera.
	}
	else {
		stream1.open(0);
		if (!stream1.isOpened())  // check if succeeded to connect to the camera
			CV_Assert("Cam open failed");
		stream1.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		stream1.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	}
	Mat cameraFrame;
	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
		return -1;
	}
	VideoWriter vidWriter;
	bool writeropened = false;
	stream1.read(cameraFrame);
	bool init = false;
	int count = 0;

	RdtInterface*r = RdtInterface::getInstance();
	Config c;
	r->init(c);

	//unconditional loop
	bool ifframe = true;
	while (ifframe) {
		ifframe = stream1.read(cameraFrame);
		if (!ifframe) {
			break;
		}

		AcceptanceStatus ret = r->process(&cameraFrame);
		Rect region_of_interest = Rect(ret.mBoundingBox.x, ret.mBoundingBox.y, ret.mBoundingBox.width, ret.mBoundingBox.height);
		cv::rectangle(cameraFrame, region_of_interest, Scalar(143, 143, 143), 3);

		imshow(window_name, cameraFrame);
		//imshow("ROI", r->mBlurrInputImage);
		//if (!init) {
		//	writeropened = vidWriter.open("out.mov", CV_FOURCC('M', 'J', 'P', 'G'), 3, cvSize(img.cols, img.rows), false);
		//	createTrackbar("Threshold", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
		//	init = true;
		//}
		if (writeropened) {
			vidWriter.write(cameraFrame);
		}
		int c, space = 0x20;
		c = waitKey(50);
		if (c == space) {
			while (1) {
				c = waitKey(50);
				if (c == space) break;
			}
		}
	}
	if (vidWriter.isOpened()) {
		vidWriter.release();
	}
	return 0;
}

