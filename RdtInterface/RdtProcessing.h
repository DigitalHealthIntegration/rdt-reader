#ifndef __RDTPROCESSING__H
#define __RDTPROCESSING__H
#include <time.h>
#include "opencv/cv.h"
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "RdtReader.h"
using namespace cv;


typedef struct {
	int64_t  mTimestamp;
	uint32_t mWidth, mHeight;
	RdtInterface mRdtInterface;
	bool mIsAcceptable;
	float mSharpness;
	float mScale;
	float mBrightness;
	float mPerspectiveDistortion;
	CvPoint mcenterOfRDT;
	//Internal need to move to internal structure to maintain.
	CvPoint *mRoiPoint;
	int32_t mMaxROIPoint;
	CvRect mROIRectangle;
} RdtStatus;

EXTERN int32_t mFrameNumber;
EXTERN RdtStatus *mRdtStatus;
EXTERN Mat mInputImage;
EXTERN Mat mGreyInput;

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

