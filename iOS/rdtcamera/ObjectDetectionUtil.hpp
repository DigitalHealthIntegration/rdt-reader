//
//  ObjectDetectionUtil.hpp
//  rdtcamera
//
//  Created by developer on 28/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

#ifndef ObjectDetectionUtil_hpp
#define ObjectDetectionUtil_hpp
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)
#define GOOD 0
#define TOO_HIGH 1
#define TOO_LOW -1

#define MAX_VALUE 10000000000000.0

static float  cannonicalArrow[]={121.0f,152.0f,182.0f};
static float  cannonicalCpattern[]={596.0f,746.0f,895.0f};
static float  cannonicalInfl[]={699.0f,874.0f,1048.0f};
static double ac_can = cannonicalCpattern[1]-cannonicalArrow[1];
static double ai_can = cannonicalInfl[1]-cannonicalArrow[1];



static CvPoint2D32f  cannonicalA_C_Mid(449.0f,30.0f);
static CvPoint2D32f  ref_A(cannonicalArrow[1]-cannonicalA_C_Mid.x,0.0f);
static CvPoint2D32f  ref_C(cannonicalCpattern[1]-cannonicalA_C_Mid.x,0.0f);
static CvPoint2D32f  ref_I(cannonicalInfl[1]-cannonicalA_C_Mid.x,0.0f);

static double ref_A_C = (cannonicalCpattern[1]-cannonicalArrow[1]);



class ObjectDetectionUtil{
    
 
    public:
         // static int REGISTRATION_LEVEL=5;
    long st;
    std::string TAG;
    int REGISTRATION_LEVEL;
        ObjectDetectionUtil(){
         st=0;
         TAG = "test";
         REGISTRATION_LEVEL=5;
        };
         cv::Mat mRefPyr;
         std::deque<std::pair<cv::Mat,cv::Mat>> mWarpList;
         cv::Mat mRefImage;
         int mRefCount=0;
         long mPreviousTime=0;
         //private short mBrightness;
         //private short mSharpness;
         cv::Mat mLocalcopy;
         bool mPlaybackMode;
         cv::Mat mPipMat;
         cv::Mat mMotionVectorMat;
         cv::Mat mWarpedMat;
         cv::Mat mWarpInfo;
         cv::Point mComputeVector_FinalPoint;
         cv::Point mComputeVector_FinalMVector;
         int checkSteady(cv::Mat greyMat);
         double detect2(CvPoint2D32f a, CvPoint2D32f c, CvPoint2D32f i, CvPoint3D32f orientations, CvPoint2D32f *out_scale_rot);
         CvPoint2D32f translate(CvPoint2D32f inp, CvPoint2D32f t);
         CvPoint2D32f warpPoint(CvPoint2D32f point,cv::Mat R);
         CvPoint2D32f swap(CvPoint2D32f xy);
         CvPoint2D32f scale(CvPoint2D32f xy, float s);
         cv::Mat makeRMat(double scale, double theta, CvPoint2D32f tr);
         static CvPoint2D32f midptPoint;
         int mMaxFrameTranslationalMagnitude = 30;
         int mMax10FrameTranslationalMagnitude =200;
        double updateTransformationMat(cv::Mat ref, cv::Mat ins,cv::Mat warpMatrix);
        cv::Mat DetectEdges(cv::Mat grayMat);
        cv::Mat FindMotion(cv::Mat inp,bool saveref);
        cv::Mat scaleAffineMat(cv::Mat warpmat, int level);
        cv::Mat getTransformation(cv::Mat ref, cv::Mat ins);
        double angleOfLine(CvPoint2D32f p1, CvPoint2D32f p2);
        cv::Mat FindMotionRefIns(cv::Mat refe,cv::Mat inp,cv::Mat warpmat ,bool resize);
        cv::Mat LaplacianCompute(cv::Mat inp);
        cv::Mat FindMotionLaplacianRefIns(cv::Mat refe,cv::Mat inp,cv::Mat warpmat ,bool resize);
        cv::Mat ComputeVector(cv::Point translation,cv::Mat m,cv::Scalar s);
        cv::Mat ComputeMotion(cv::Mat greyMat);
        //        long st;
        //        std::string TAG;
        //        int REGISTRATION_LEVEL;
        double lengthOfLine(CvPoint2D32f p1, CvPoint2D32f p2);
        bool angle_constraint(double orientation, double theta_deg);
        cv::Mat GetTransform(cv::Mat refM, cv::Mat insM);
};

#endif  
