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
#define MAX_VALUE 10000000000000.0

static float  cannonicalArrow[]={121.0f,152.0f,182.0f};
static float  cannonicalCpattern[]={596.0f,746.0f,895.0f};
static float  cannonicalInfl[]={699.0f,874.0f,1048.0f};
static double ac_can = cannonicalCpattern[1]-cannonicalArrow[1];
static double ai_can = cannonicalInfl[1]-cannonicalArrow[1];
static CvPoint2D32f midptPoint;


static CvPoint2D32f  cannonicalA_C_Mid(449.0f,30.0f);
static CvPoint2D32f  ref_A(cannonicalArrow[1]-cannonicalA_C_Mid.x,0.0f);
static CvPoint2D32f  ref_C(cannonicalCpattern[1]-cannonicalA_C_Mid.x,0.0f);
static CvPoint2D32f  ref_I(cannonicalInfl[1]-cannonicalA_C_Mid.x,0.0f);

static double ref_A_C = (cannonicalCpattern[1]-cannonicalArrow[1]);


double detect2(CvPoint2D32f a, CvPoint2D32f c, CvPoint2D32f i, CvPoint3D32f orientations, CvPoint2D32f *out_scale_rot);
CvPoint2D32f translate(CvPoint2D32f inp, CvPoint2D32f t);
CvPoint2D32f warpPoint(CvPoint2D32f point,cv::Mat R);
CvPoint2D32f swap(CvPoint2D32f xy);
CvPoint2D32f scale(CvPoint2D32f xy, float s);
cv::Mat makeRMat(double scale, double theta, CvPoint2D32f tr);


#endif  
