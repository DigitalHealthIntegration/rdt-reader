//
//  ObjectDetectionUtil.cpp
//  rdtcamera
//
//  Created by developer on 28/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

#include "ObjectDetectionUtil.hpp"
#include <math.h>
static double angleOfLine(CvPoint2D32f p1, CvPoint2D32f p2)
{
    
    return atan2((p2.y-p1.y),(p2.x-p1.x));
}

CvPoint2D32f warpPoint(CvPoint2D32f point,cv::Mat R)
{
    cv::Point result;

    result.x = point.x * R.at<double>(0,0,0) + point.y *  R.at<double>(0,1,0)+  R.at<double>(0,2,0);
    result.y = point.x * R.at<double>(1,0,0) + point.y *  R.at<double>(1,1,0)+  R.at<double>(1,2,0);
    return result;
}

CvPoint2D32f translate(CvPoint2D32f inp, CvPoint2D32f t)
{   CvPoint2D32f res;
    res.x=inp.x+t.x;
    res.y=inp.y+t.y;
    return res;
}
static double lengthOfLine(CvPoint2D32f p1, CvPoint2D32f p2){
     return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
 }
CvPoint2D32f swap(CvPoint2D32f xy)
{
    return CvPoint2D32f(xy.y,xy.x);
}
CvPoint2D32f scale(CvPoint2D32f xy, float s)
{
    return CvPoint2D32f(xy.x *s,xy.y *s);
}
static bool angle_constraint(double orientation, double theta_deg) {
       double T=30.0;
       double d=abs(orientation-theta_deg);
       if(d>180) d=360.0-d;
       if(d>T) return true;
       return false;
   }

cv::Mat makeRMat(double scale, double theta, CvPoint2D32f tr)
   {
       double cos_th=cos(theta);
       double sin_th=sin(theta);
       cv::Mat R = (cv::Mat_<double>(2,3) << cos_th*scale, 0-sin_th*scale, tr.x, sin_th*scale, cos_th*scale, tr.y);

//       R.put(0,0,cos_th*scale);R.put(0,1,0-sin_th*scale);R.put(0,2,tr.x);
//       R.put(1,0,sin_th*scale);R.put(1,1,cos_th*scale);R.put(1,2,tr.y);
       
       return R;
   }

double detect2(CvPoint2D32f a, CvPoint2D32f c, CvPoint2D32f i, CvPoint3D32f orientations, CvPoint2D32f *out_scale_rot)
{
    //rotation
    double th1=angleOfLine(a,c);
    double th2=angleOfLine(a,i);
    double theta=(th1+th2)/2;
    if(theta<0) theta+=2*M_PI;

    //avoid feature orientations which are very different from theta
    
    double theta_deg=RADIANS_TO_DEGREES(theta);
    if(angle_constraint(orientations.x,theta_deg)||angle_constraint(orientations.y,theta_deg) ||angle_constraint(orientations.z,theta_deg))
    {
        return MAX_VALUE;
    }

    //scale
    double ac=lengthOfLine(a,c);
    double ai=lengthOfLine(a,i);

    double s1=ac/ac_can;
    double s2=ai/ai_can;
    double scale=sqrt(s1*s2);

    //avoid scales which are very different from each other
    double scale_disparity=s1/s2;
    if(scale_disparity>1.25 || scale_disparity<0.75)
    {
        return MAX_VALUE;
    }

    //The inspection points rotate back so use -theta angle
    double cos_th=cos(-1*theta);
    double sin_th=sin(-1*theta);

    cv::Mat R = (cv::Mat_<double>(2,3) << cos_th/scale, 0-sin_th/scale, 0, sin_th/scale, cos_th/scale, 0);
//    R.put(0,0,cos_th/scale);R.put(0,1,0-sin_th/scale);R.put(0,2,0);
//    R.put(1,0,sin_th/scale);R.put(1,1,cos_th/scale);R.put(1,2,0);

    //Now warp the points
    CvPoint2D32f a1=warpPoint(a,R);
    CvPoint2D32f c1=warpPoint(c,R);
    CvPoint2D32f i1=warpPoint(i,R);

    CvPoint2D32f ac1_mid((a1.x+c1.x)/2,(a1.y+c1.y)/2);
    //translate back to 0,0
    a1=CvPoint2D32f(a1.x-ac1_mid.x,a1.y-ac1_mid.y);
    c1=CvPoint2D32f(c1.x-ac1_mid.x,c1.y-ac1_mid.y);
    i1=CvPoint2D32f(i1.x-ac1_mid.x,i1.y-ac1_mid.y);

    out_scale_rot->x=scale;
    out_scale_rot->y=theta;
//    R.release();
    
    
    //compute the MSE
    return (lengthOfLine(ref_A,a1)+lengthOfLine(ref_C,c1)+lengthOfLine(ref_I,i1))/3;
}
