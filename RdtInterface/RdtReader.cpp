#include <malloc.h>
//#include "boost/date_time/posix_time/posix_time.hpp"

//#include <boost/chrono.hpp>

#ifndef EXTERN
#define EXTERN 
#include "RdtProcessing.h"
#endif

//using namespace boost::posix_time;
using namespace cv;

void init() {
	LOGIT(__func__);
	mFrameNumber = 0;
	if (NULL == mRdtStatus) {
		mRdtStatus = (RdtStatus*)malloc(1*sizeof(RdtStatus));
		mRdtStatus->mTimestamp = 0;
		mRdtStatus->mHeight = 0;
		mRdtStatus->mWidth = 0;
		SetDefaultRdtInterface(&mRdtStatus->mRdtInterface);
		mRdtStatus->mMaxROIPoint = 10;
		mRdtStatus->mRoiPoint = (CvPoint*) malloc(mRdtStatus->mMaxROIPoint * sizeof(CvPoint));
	}else {
		freeMemory();
	}
}

RdtInterface update(void* ptr) {
	Mat mInputImage = Mat(mRdtStatus->mWidth, mRdtStatus->mHeight, CV_8UC3, ptr);
	LOGIT(__func__);
	if (mRdtStatus) {
		mRdtStatus->mTimestamp = getTime();
	}
	mFrameNumber++;
	RdtInterface r;
	SetDefaultRdtInterface(&r);
	r.mAcceptanceStatus = mRdtStatus->mRdtInterface.mAcceptanceStatus;

	r.mTimestamp = mFrameNumber;
	return r;
}

void term() {
	LOGIT(__func__);
	freeMemory();
}

void setDefaultStatus(AcceptanceStatus *as){
	as->bRDTFound = false;
	as->brightness = NOT_COMPUTED;
	as->sharpness = NOT_COMPUTED;
	as->scale = NOT_COMPUTED;
	as->displacementX = NOT_COMPUTED;
	as->displacementY = NOT_COMPUTED;
	as->bRDTFound = false;
}

void SetDefaultRdtInterface(RdtInterface* rdtInt) {
	rdtInt->mBrightness = -1;
	rdtInt->mSharpness = -1;
	rdtInt->mPerspectiveDistortion = -1.0;
	rdtInt->mScale = -1;
	rdtInt->mTimestamp = 0;
	setDefaultStatus(&rdtInt->mAcceptanceStatus);
}

void freeMemory() {
	if (mRdtStatus) {
		if (mRdtStatus->mRoiPoint) {
			free(mRdtStatus->mRoiPoint);
		}
		mRdtStatus->mRoiPoint = NULL;
		free(mRdtStatus);
	}
	mRdtStatus = NULL;
}

void convertInputImageToGrey() {
	/// Convert the image to grayscale
	cvtColor(mInputImage, mGreyInput, CV_BGR2GRAY);
}

bool coarseROIFinder() {
	for (int i = 0; i < mRdtStatus->mMaxROIPoint; i++) {
		mRdtStatus->mRoiPoint[i] = cvPoint(i, i);
	}
	return 	true;
}

bool checkForROI() {

	return true;
}
bool computeROIRectangle() {

	mRdtStatus->mROIRectangle=cvRect(0, 0, 10, 10);
	return true;
}

bool computeBlurr(Mat input) {
	Mat laplacian;
	Laplacian(input,laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
	cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
	meanStdDev(laplacian, mean, stddev, cv::Mat());
	mRdtStatus->mSharpness = stddev.val[0] * stddev.val[0];
	return true;
}

bool computeBrightness(Mat input) {
	cv:Scalar tempVal = cv::mean(input);
	mRdtStatus->mBrightness = tempVal.val[0];
	return true;
}

int64_t getTime() {
//	boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
//	int64_t ms = (boost::posix_time::microsec_clock::local_time() - time_epoch).total_microseconds();
	return 0;
}
