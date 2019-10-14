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
//	cv::imwrite("./img.png", mInputImage);
	if(mInputImage.channels() == 3) {
		cvtColor(mInputImage, mGreyInput, CV_BGR2GRAY);
	}else if(mInputImage.channels() == 4) {
		cvtColor(mInputImage, mGreyInput, CV_BGRA2GRAY);
	}
}

bool RdtInterface::computeDistortion(AcceptanceStatus& status){
	if(mROIRectangle.width > mConf.mMaxScale){
		status.mScale = TOO_HIGH;
		return false;
	}else if (mROIRectangle.width < mConf.mMinScale){
		status.mScale = TOO_LOW;
		return false;
	}else status.mScale = GOOD;

	if (mROIRectangle.x > mConf.mXMax){
		status.mDisplacementX = TOO_HIGH;
		return false;
	}else if (mROIRectangle.x < mConf.mXMin){
		status.mDisplacementX = TOO_LOW;
		return false;
	}else status.mDisplacementX = GOOD;

	if (mROIRectangle.y > mConf.mYMax){
		status.mDisplacementY = TOO_HIGH;
		return false;
	}else if (mROIRectangle.y < mConf.mYMin){
		status.mDisplacementY = TOO_LOW;
		return false;
	}else status.mDisplacementY = GOOD;
	status.mPerspectiveDistortion= GOOD;
	return true;
}

bool RdtInterface::computeROIRectangle() {
	//randomly return false
	mROIRectangle = cvRect(150, 300, 900,45);
	return true;
}


bool RdtInterface::computeBlur(AcceptanceStatus& status) {
	Mat laplacian;
	Laplacian(mGreyInput, laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
	cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
	meanStdDev(laplacian, mean, stddev, cv::Mat());
	mSharpness = stddev.val[0] * stddev.val[0];
	if (mSharpness < mConf.mMinSharpness){
		status.mSharpness = TOO_LOW;
		return false;
	}
	status.mSharpness = GOOD;
	return true;
}
bool RdtInterface::computeBrightness(AcceptanceStatus& status) {
	cv:Scalar tempVal = cv::mean(mGreyInput);
	mBrightness = tempVal.val[0];
	if (mBrightness > mConf.mMaxBrightness) {
		status.mBrightness = TOO_HIGH;
		return false;
	}else if (mBrightness < mConf.mMinBrightness){
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
	cv::Mat* pInputImage = (cv::Mat*)imagePtr;
	mInputImage = pInputImage->clone();
	convertInputImageToGrey();
	ret.mRDTFound=computeROIRectangle();
	if (ret.mRDTFound){
		ret.mBoundingBox.x = mROIRectangle.x;
		ret.mBoundingBox.y = mROIRectangle.y;
		ret.mBoundingBox.width = mROIRectangle.width;
		ret.mBoundingBox.height = mROIRectangle.height;
	}else return ret;

	if (!computeDistortion(ret)){
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

void RdtInterface::setConfig(Config c) {
	mConf = c;
}


bool RdtInterface::init(Config c){
	RdtInterface::getInstance()->setConfig(c);
    RdtInterface::getInstance()->initialize();
    return true;
}

AcceptanceStatus update(void *ptr){
	AcceptanceStatus ret = RdtInterface::getInstance()->process(ptr);
    return ret; ;
}
void term() {
	//RdtInterface::getInsta
}

