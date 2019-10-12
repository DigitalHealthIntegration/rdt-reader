#include "RdtProcessing.h"

AcceptanceStatus::AcceptanceStatus() {
    setDefaultStatus(this);
}
AcceptanceStatus::~AcceptanceStatus(){
}

void AcceptanceStatus::setDefaultStatus(AcceptanceStatus *as){
    as->mRDTFound = false;
    as->mBrightness = NOT_COMPUTED;
    as->mSharpness = NOT_COMPUTED;
    as->mScale = NOT_COMPUTED;
    as->mDisplacementX = NOT_COMPUTED;
    as->mDisplacementY = NOT_COMPUTED;
    as->mRDTFound = false;
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

bool RdtInterface::coarseROIFinder() {
	//for (int i = 0; i < mRdtStatus->mMaxROIPoint; i++) {
	//	mRdtStatus->mRoiPoint[i] = cvPoint(i, i);
	//}
	return 	true;
}

bool RdtInterface::checkForROI() {

	return true;
}
bool RdtInterface::computeROIRectangle() {
	mROIRectangle = cvRect(0, 0, 10, 10);
	return true;
}

bool RdtInterface::computeBlurr() {
	Mat laplacian;
	Laplacian(mGreyInput, laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
	cv::Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
	meanStdDev(laplacian, mean, stddev, cv::Mat());
	mSharpness = stddev.val[0] * stddev.val[0];
	return true;
}

bool RdtInterface::computeBrightness() {
	cv:Scalar tempVal = cv::mean(mGreyInput);
	mBrightness = tempVal.val[0];
	return true;
}

void RdtInterface::process(void *imagePtr){

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

