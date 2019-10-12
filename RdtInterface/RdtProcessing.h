#ifndef __RDTPROCESSING__H
#define __RDTPROCESSING__H
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "opencv/cv.h"
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "RdtReader.h"


using namespace cv;

#define LOGIT(x) {\
	printf("%s\n",x);\
}



class RdtInterface {
public:
	static RdtInterface* getInstance();
	~RdtInterface();
	void initialize();
	void process(void* ptr);
	void term();
	AcceptanceStatus getAcceptanceStatus();
private:
	static RdtInterface *mRdtInterface;
	void setDefaults();
	RdtInterface();
	bool checkForROI();
	void convertInputImageToGrey();
	bool coarseROIFinder();
	bool computeROIRectangle();
	bool computeBlurr();
	bool computeBrightness();


	int64_t  mTimestamp;
	AcceptanceStatus mAcceptanceStatus;
	bool mIsAcceptable;
	float mSharpness;
	float mScale;
	float mBrightness;
	float mPerspectiveDistortion;
	Mat mInputImage;
	Mat mGreyInput;
	uint32_t mWidth, mHeight;
	CvPoint *mRoiPoint;
	int32_t mMaxROIPoint;
	CvRect mROIRectangle;
};


void setDefaultStatus(AcceptanceStatus *as);
void freeMemory();
int64_t getTime();
bool checkForROI();
void convertInputImageToGrey();
bool coarseROIFinder();
bool computeROIRectangle();
bool computeBlurr();
bool computeBrightness();
void SetDefaultRdtInterface(RdtInterface *rdtInterface);

#endif // !__RDTPROCESSING__H

