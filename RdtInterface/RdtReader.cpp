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
		mRdtStatus->mRdtInterface.mBrightness = -1;
		mRdtStatus->mRdtInterface.mSharpness = -1;
		mRdtStatus->mRdtInterface.mPerspectiveDistortion = - 1.0;
		mRdtStatus->mRdtInterface.mScale = -1;
		mRdtStatus->mRdtInterface.mTimestamp = 0;
		setDefaultStatus(&mRdtStatus->mRdtInterface.mAcceptanceStatus);
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
	setDefaultStatus(&r.mAcceptanceStatus);
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

bool computeBlurr() {

	return true;
}

bool computeBrightness() {

	return true;
}

int64_t getTime() {
//	boost::posix_time::ptime const time_epoch(boost::gregorian::date(1970, 1, 1));
//	int64_t ms = (boost::posix_time::microsec_clock::local_time() - time_epoch).total_microseconds();
	return 0;
}
