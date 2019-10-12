#include "RdtProcessing.h"

AcceptanceStatus::AcceptanceStatus() {
    setDefaultStatus();
}
AcceptanceStatus::~AcceptanceStatus(){
}

void AcceptanceStatus::setDefaultStatus(){
    mRDTFound = false;
	mBrightness = NOT_COMPUTED;
    mSharpness = NOT_COMPUTED;
    mScale = NOT_COMPUTED;
    mDisplacementX = NOT_COMPUTED;
    mDisplacementY = NOT_COMPUTED;
	mPerspectiveDistortion = NOT_COMPUTED;
}

bool AcceptanceStatus::getRdtFound() {
	return mRDTFound;
}

RdtInterface* (RdtInterface::mRdtInterface) = NULL;

RdtInterface::RdtInterface(){

}
RdtInterface::~RdtInterface(){

}

RdtInterface* RdtInterface::getInstance() {
    if(mRdtInterface == NULL){
        mRdtInterface= new RdtInterface();
    }
    return mRdtInterface;
}

AcceptanceStatus RdtInterface::getAcceptanceStatus() {
	return mAcceptanceStatus;
}

void RdtInterface::setDefaults() {
	mBrightness = -1;
	mSharpness = -1;
	mPerspectiveDistortion = -1.0;
	mScale = -1;
	mTimestamp = 0;
}

void RdtInterface::initialize() {
    setDefaults();
}


void RdtInterface::convertInputImageToGrey() {
	/// Convert the image to grayscale
	cvtColor(mInputImage, mGreyInput, CV_BGR2GRAY);
}

#define SCALE_MAX 1100
#define SCALE_MIN 700

#define X_MIN 100
#define X_MAX 500

#define Y_MIN 50
#define Y_MAX 650

bool RdtInterface::computeDistortion(AcceptanceStatus& status)
{
	if(mROIRectangle.width > SCALE_MAX)
	{
		status.mScale = TOO_HIGH;
		return false;
	}
	else if (mROIRectangle.width < SCALE_MIN)
	{
		status.mScale = TOO_LOW;
		return false;
	}
	else status.mScale = GOOD;

	if (mROIRectangle.x > X_MAX)
	{
		status.mDisplacementX = TOO_HIGH;
		return false;
	}
	else if (mROIRectangle.x < X_MIN)
	{
		status.mDisplacementX = TOO_LOW;
		return false;
	}
	else status.mDisplacementX = GOOD;

	if (mROIRectangle.y > Y_MAX)
	{
		status.mDisplacementY = TOO_HIGH;
		return false;
	}
	else if (mROIRectangle.y < Y_MIN)
	{
		status.mDisplacementY = TOO_LOW;
		return false;
	}
	else status.mDisplacementY = GOOD;
	
	status.mPerspectiveDistortion= GOOD;
	//Todo add logic for distortion 

	return true;
}
bool RdtInterface::computeROIRectangle() {
	//randomly return false

	mROIRectangle = cvRect(150, 300, 900,45);
	
	return true;
}

#define SHARP_MIN 500.0f

bool RdtInterface::computeBlur(AcceptanceStatus& status) {
	Mat laplacian;
	Laplacian(mGreyInput, laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
	cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
	meanStdDev(laplacian, mean, stddev, cv::Mat());
	mSharpness = stddev.val[0] * stddev.val[0];

	if (mSharpness < SHARP_MIN)
	{
		status.mSharpness = TOO_LOW;
		return false;
	}
	status.mSharpness = GOOD;

	return true;
}
#define BRIGHT_MAX 210.0f
#define BRIGHT_MIN 160.0f
bool RdtInterface::computeBrightness(AcceptanceStatus& status) {
	cv:Scalar tempVal = cv::mean(mGreyInput);
	mBrightness = tempVal.val[0];
	if (mBrightness > BRIGHT_MAX) {
		status.mBrightness = TOO_HIGH;
		return false;
	}
	else if (mBrightness < BRIGHT_MIN)
	{
		status.mBrightness = TOO_LOW;
		return false;
	}
	status.mBrightness = GOOD;
	return true;
}

AcceptanceStatus RdtInterface::process(void *imagePtr){
	AcceptanceStatus ret;
	if (imagePtr == NULL)
		return ret;
	convertInputImageToGrey();
	ret.mRDTFound=computeROIRectangle();
	if (ret.mRDTFound)
	{
		ret.mBoundingBox.x = mROIRectangle.x;
		ret.mBoundingBox.y = mROIRectangle.y;
		ret.mBoundingBox.width = mROIRectangle.width;
		ret.mBoundingBox.height = mROIRectangle.height;
	}
	else return ret;

	if (!computeDistortion(ret))
	{
		return ret;
	}

	if (!computeBrightness(ret)) 
		return ret;

	if (!computeBlur(ret))
		return ret;

	return ret;
}

void RdtInterface::term() {

}

bool init(){
    RdtInterface::getInstance()->initialize();
    return true;
}

void update(void *ptr){
    RdtInterface::getInstance()->process(ptr);
    return ;
}
void term() {
	//RdtInterface::getInsta
}

