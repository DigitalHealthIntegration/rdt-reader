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
#include "rdtReader.h"


using namespace cv;

class RdtInterface {
public:
	static RdtInterface* getInstance();
	~RdtInterface();
	void initialize();
	AcceptanceStatus process(void* ptr);
	void term();
	AcceptanceStatus getAcceptanceStatus();
	bool init(Config c);

private:
	static RdtInterface *mRdtInterface;
	void setDefaults();
	RdtInterface();
	void setConfig(Config c);
	void convertInputImageToGrey();
	bool computeROIRectangle();
	bool computeBlur(AcceptanceStatus& status);
	bool computeBrightness(AcceptanceStatus& status);
	bool computeDistortion(AcceptanceStatus& status);

	int64_t  mTimestamp;
	AcceptanceStatus mAcceptanceStatus;
	Config mConf;
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

#endif // !__RDTPROCESSING__H

