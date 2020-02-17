//
//  ObjectDetectionUtil.cpp
//  rdtcamera
//
//  Created by developer on 28/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

#include "ObjectDetectionUtil.hpp"
#include <math.h>


double ObjectDetectionUtil::angleOfLine(CvPoint2D32f p1, CvPoint2D32f p2)
{
    
    return atan2((p2.y-p1.y),(p2.x-p1.x));
}

CvPoint2D32f ObjectDetectionUtil::warpPoint(CvPoint2D32f point,cv::Mat R)
{
    cv::Point result;

    result.x = point.x * R.at<double>(0,0,0) + point.y *  R.at<double>(0,1,0)+  R.at<double>(0,2,0);
    result.y = point.x * R.at<double>(1,0,0) + point.y *  R.at<double>(1,1,0)+  R.at<double>(1,2,0);
    return result;
}

CvPoint2D32f ObjectDetectionUtil::translate(CvPoint2D32f inp, CvPoint2D32f t)
{   CvPoint2D32f res;
    res.x=inp.x+t.x;
    res.y=inp.y+t.y;
    return res;
}
double ObjectDetectionUtil::lengthOfLine(CvPoint2D32f p1, CvPoint2D32f p2){
     return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
 }
CvPoint2D32f ObjectDetectionUtil::swap(CvPoint2D32f xy)
{
    return CvPoint2D32f(xy.y,xy.x);
}
CvPoint2D32f ObjectDetectionUtil::scale(CvPoint2D32f xy, float s)
{
    return CvPoint2D32f(xy.x *s,xy.y *s);
}
bool ObjectDetectionUtil::angle_constraint(double orientation, double theta_deg) {
       double T=30.0;
       double d=abs(orientation-theta_deg);
       if(d>180) d=360.0-d;
       if(d>T) return true;
       return false;
   }

cv::Mat ObjectDetectionUtil::makeRMat(double scale, double theta, CvPoint2D32f tr)
   {
       double cos_th=cos(theta);
       double sin_th=sin(theta);
       cv::Mat R = (cv::Mat_<double>(2,3) << cos_th*scale, 0-sin_th*scale, tr.x, sin_th*scale, cos_th*scale, tr.y);

//       R.put(0,0,cos_th*scale);R.put(0,1,0-sin_th*scale);R.put(0,2,tr.x);
//       R.put(1,0,sin_th*scale);R.put(1,1,cos_th*scale);R.put(1,2,tr.y);
       
       return R;
   }

double ObjectDetectionUtil::detect2(CvPoint2D32f a, CvPoint2D32f c, CvPoint2D32f i, CvPoint3D32f orientations, CvPoint2D32f *out_scale_rot)
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
cv::Mat ObjectDetectionUtil::scaleAffineMat(cv::Mat warpmat, int level) {
    cv::Mat warp= warpmat.clone();
    int factor = 1<<level;
    warp.at<float>(0,2) = warp.at<float>(0,2)*factor;
    warp.at<float>(1,2) = warp.at<float>(1,2)*factor;
    return warp;
   }
cv::Mat ObjectDetectionUtil::ComputeVector(cv::Point translation,cv::Mat m,cv::Scalar s) {
        double y = translation.y;//warp.get(1, 2)[0];
        double x = translation.x;//warp.get(0, 2)[0];
        double r = sqrt(x * x + y * y);

        double angleRadian = atan2(y, x);
        if(angleRadian < 0){
            angleRadian += M_PI * 2;;
        }
//        Log.d("ComputedAngle", r+"["+Math.toDegrees(angleRadian) +"]");
//        if (x < 0.0) { //2  and 3 quad
//            angleRadian = angleRadian + Math.PI;
//        } else if (x >= 0.0 && y < 0.0) {
//            angleRadian = angleRadian + Math.PI * 2;
//        }
        double x1 = abs(r * cos(angleRadian));
        double y1 = abs(r * sin(angleRadian));
        double angle = RADIANS_TO_DEGREES(angleRadian);
        if( angle>=0 && angle <=90){
            x1 = 100+x1;
            y1 = 100-y1;
        }else if (angle > 90 && angle <= 180){
            x1 = 100-x1;
            y1 = 100-y1;
        }else if (angle > 180 && angle <= 270) {
            x1 = 100-x1;
            y1 = 100+y1;
        }else if(angle >270 && angle <=360){
            x1 = 100+x1;
            y1 = 100+y1;
        }
        cv::Point p;
//        Log.d("Points", "[100,100] -> ["+x1+","+y1+"]");
        if(sizeof m != 0) {
            m = cv::Mat(200, 200, CV_8UC4);
            m.setTo(cv::Scalar(0));
        }
        cv::line(m, cv::Point(100,100), cv::Point(x1,y1),s,5);
        mComputeVector_FinalPoint=p;
        mComputeVector_FinalMVector = cv::Point(r,RADIANS_TO_DEGREES(angleRadian));
        return m;
    }
    cv::Mat ObjectDetectionUtil::GetTransform(cv::Mat refM, cv::Mat insM) {
        cv::Mat ref ;
        pyrDown(refM, ref);
        cv::Mat ins ;
        pyrDown(insM, ins);
        cv::Mat warpMatrix = getTransformation(ref, ins);
        return warpMatrix;
    }
//
    cv::Mat ObjectDetectionUtil::FindMotion(cv::Mat inp,bool saveref) {
        cv::Mat ins;
        pyrDown(inp, ins);

        for(int i=0;i<REGISTRATION_LEVEL-1;i++){
            pyrDown(ins, ins);
        }
        cv::Mat warpMatrix;
        if(mRefPyr.data!= nullptr) {
            warpMatrix = getTransformation(ins,mRefPyr);
        }
        if(saveref)mRefPyr = ins.clone();
        return warpMatrix;
    }
//
cv::Mat ObjectDetectionUtil::FindMotionRefIns(cv::Mat refe,cv::Mat inp,cv::Mat warpmat ,bool resize){
        cv::Mat ins;
        cv::Mat ref;
        if(resize){
            cv::Size s(inp.cols>>REGISTRATION_LEVEL,inp.rows>>REGISTRATION_LEVEL);
            ins = cv::Mat((int)s.width,(int)s.height,inp.type());
            ref = cv::Mat((int)s.width,(int)s.height,inp.type());
            cv::resize(inp,ins,s, 0.0, 0.0, cv::INTER_CUBIC);
            cv::resize(refe,ref,s, 0.0, 0.0, cv::INTER_CUBIC);
        }else {
            pyrDown(inp, ins);
            for (int i = 0; i < REGISTRATION_LEVEL - 1; i++) {
                pyrDown(ins, ins);
            }
            pyrDown(refe, ref);
            for (int i = 0; i < REGISTRATION_LEVEL - 1; i++) {
                pyrDown(ref, ref);
            }
        }
        cv::Mat warp;
        double ret  = updateTransformationMat(DetectEdges(ins),DetectEdges(ref),warpmat);
        if (ret >0.0) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            //ComputeVector
        }else{
            warp = cv::Mat::eye(2,3,CV_32F);
            warpmat=warp.clone();
            warp.at<float>(0,2) = refe.cols;
            warp.at<float>(1,2) = refe.rows;
        }
        ref.release();
        ins.release();
        return warp;
    }
//
//
//
//
cv::Mat ObjectDetectionUtil::LaplacianCompute(cv::Mat inp){
    cv::Mat ins1;
    pyrDown(inp, ins1);
    cv::Mat up(inp.cols,inp.rows,inp.type());
    pyrUp(ins1,up);
    cv::Mat lap(inp.cols,inp.rows,CV_16S);
    //Core.addWeighted(inp,1.0,up,-1.0,255,lap,lap.type());
    //convertScaleAbs(lap,up);
    cv::subtract(inp,up,lap);
    convertScaleAbs(lap, up);
//        SaveMatrix(up,"insup");
    return up;
    }
cv::Mat ObjectDetectionUtil::FindMotionLaplacianRefIns(cv::Mat refe,cv::Mat inp,cv::Mat warpmat ,bool resize){
        cv::Mat ins;
        cv::Mat ref;
        cv::Mat ins1;

        int w=0,h=0;
        if(resize){
            cv::Size s(inp.cols>>REGISTRATION_LEVEL,inp.rows>>REGISTRATION_LEVEL);
            ins = cv::Mat((int)s.width,(int)s.height,inp.type());
            ref = cv::Mat((int)s.width,(int)s.height,inp.type());
            cv::resize(inp,ins,s, 0.0, 0.0, cv::INTER_CUBIC);
            cv::resize(refe,ref,s, 0.0, 0.0, cv::INTER_CUBIC);
        }else {
            pyrDown(inp, ins);
            w = ins.rows;
            h = ins.cols;
            for (int i = 0; i < REGISTRATION_LEVEL-1 ; i++) {
                w = ins.rows;
                h = ins.cols;
                pyrDown(ins, ins);
            }

//            ins = LaplacianCompute(ins);
//
            int kernel_size = 3;
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            cv::Mat temp;
//            Laplacian( ins, temp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
//            convertScaleAbs( temp, ins,1,0);

            pyrDown(ins, ins1);
            cv::Mat up(w,h,ins.type());
            pyrUp(ins1,up);
//            //SaveMatrix(up,"insup");
            cv::subtract(ins,up,ins);
//            //ins = ins - up;

            pyrDown(refe, ref);
            w = ref.rows;
            h = ref.cols;
            for (int i = 0; i < REGISTRATION_LEVEL-1 ; i++) {
                w = ref.rows;
                h = ref.cols;
                pyrDown(ref, ref);
            }
//            ref = LaplacianCompute(ref);

//            Laplacian( ref, temp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
//            convertScaleAbs( temp, ref,1,0);

            pyrDown(ref, ins1);
            up=cv::Mat(w,h,ref.type());
            pyrUp(ins1,up);
            //SaveMatrix(up,"refup");
            cv::subtract(ref,up,ref);
//            Mat dst = new Mat();
//            Core.convertScaleAbs(ins,dst,1,128);
//            SaveMatrix(dst,"ref");

        }
        cv::Mat warp;
        double ret  = updateTransformationMat(ins,ref,warpmat);
        if (ret >0.0) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            //ComputeVector
        }else{
            warp = cv::Mat::eye(2,3,CV_32F);
            warpmat=warp.clone();
            warp.at<float>(0,2) = refe.cols;
            warp.at<float>(1,2) = refe.rows;
        }
        ref.release();
        ins.release();
        return warp;
    }
//
    cv::Mat ObjectDetectionUtil::ComputeMotion(cv::Mat greyMat) {
        cv::Mat warp;
        cv::Mat warpmat = FindMotion(greyMat, true);
        if (warpmat.data != nullptr) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            //ComputeVector
        }else{
            warp = cv::Mat::eye(2,3,CV_32F);
            warp.at<float>(0,0)=1.0;
            warp.at<float>(1,1)=1.0;
            warp.at<float>(0,2)=greyMat.cols;
            warp.at<float>(1,2)=greyMat.rows;

//            warp.put(0,0,1.0);
//            warp.put(1,1,1.0);
//            warp.put(0,2,greyMat.width());
//            warp.put(1,2,greyMat.height());
        }
        return warp;
    }

    double ObjectDetectionUtil::updateTransformationMat(cv::Mat ref, cv::Mat ins,cv::Mat warpMatrix) {
        //SaveMatrix(ref,"ins");
        //SaveMatrix(ins,"ref");
        // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        int warp_mode = cv::MOTION_TRANSLATION;
        double ret = -1.0;
        try {
            int numIter = 50;
            double terminationEps = 1e-3;
            cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, numIter, terminationEps);
            ret = cv::findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, cv::Mat());
        }catch(cv::Exception e){
            return -1.0;
        }
        return ret;
    }

    cv::Mat ObjectDetectionUtil::getTransformation(cv::Mat ref, cv::Mat ins) {
       // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        int warp_mode = cv::MOTION_TRANSLATION;
        cv::Mat warpMatrix = cv::Mat::eye(2,3,CV_32F);
        cv::Mat defResp;
        try {
            int numIter = 50;
            double terminationEps = 1e-3;
            cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, numIter, terminationEps);
            double r= cv::findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, cv::Mat());
            if(r == -1){
                return defResp;
            }
        }catch(cv::Exception e){
            return defResp;
        }
        return warpMatrix;
    }
    cv::Mat ObjectDetectionUtil::DetectEdges(cv::Mat grayMat){
        //Matrices to store gradient and absolute gradient respectively
        cv::Mat grad_x;
        cv::Mat abs_grad_x;

        cv::Mat grad_y;
        cv::Mat abs_grad_y;
        //Calculating gradient in horizontal direction
        cv::Sobel(grayMat, grad_x, CV_16S, 1, 0, 3, 1, 0);

        //Calculating gradient in vertical direction
        cv::Sobel(grayMat, grad_y, CV_16S, 0, 1, 3, 1, 0);

        //Calculating absolute value of gradients in both the direction
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);

        //Calculating the resultant gradient
        cv::Mat sobel; //Mat to store the final result
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 1, sobel);

        grad_x.release();
        abs_grad_x.release();
        grad_y.release();
        abs_grad_y.release();

        return sobel;
    }


int ObjectDetectionUtil::checkBrightness(cv::Mat greyMat){
    cv::Scalar tempVal = mean(greyMat);
    double brightness = tempVal.val[0];
    
    return (int) brightness;
}

int ObjectDetectionUtil::checkSteady(cv::Mat greyMat){
    int res = 0;
    cv::Point lt;
    cv::Point rb;
    cv::Mat warp;

    warp = ComputeMotion(greyMat);
    std::pair<cv::Mat,cv::Mat> item(warp.clone(),greyMat.clone());
    mWarpList.push_back(item);
    
    long stend = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count()-st;
    if(mWarpList.size() >10){
        mWarpList.pop_front();
    }
    //Lets Check Motion now.
    if(sizeof mRefImage != 0) {
        mRefImage = greyMat.clone();
        mWarpInfo = cv::Mat::eye(2,3,CV_32F);
    }
    cv::Point p(0,0);
    for(int i=1;i<mWarpList.size();i++){
        p.x +=mWarpList[i].first.at<float>(0,2);
        p.y +=mWarpList[i].first.at<float>(1,2);
        //Log.i(i+"", mWarpList.elementAt(i).get(0,2)[0] + "x" + mWarpList.elementAt(i).get(1,2)[0]);
    }
    cv::Mat warp10 = FindMotionRefIns(mRefImage,greyMat,mWarpInfo,false);
    mWarpedMat = cv::Mat(greyMat.cols, greyMat.rows, greyMat.type());
    warpAffine(greyMat,mWarpedMat,warp10,greyMat.size());
    cv::resize(mWarpedMat, mWarpedMat, cv::Size(mWarpedMat.cols>>2,mWarpedMat.rows>>2), 0.0, 0.0, cv::INTER_CUBIC);
//

    long currtime  = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
    if((currtime - mPreviousTime > 2*1000)/*||(Math.abs(warp10.get(0,2)[0]) >= (greyMat.width()>>2))||(Math.abs(warp10.get(1,2)[0]) >= (greyMat.height()>>2))*/){
        std::cout<<"ref image updated\n\n";
        mRefCount = 0;
        mPreviousTime=currtime;
//        mRefImage.release();
//        mWarpInfo.release();
        mRefImage = greyMat.clone();
        mWarpInfo = cv::Mat::eye(2,3,CV_32F);
        cv::putText(mWarpedMat, "RDT REFERENCE IMAGE ", cv::Point(0, mWarpedMat.cols>>1), cv::FONT_HERSHEY_SIMPLEX, 2.0,cv::Scalar(255,0,0,0),5);
    }

    //Threshold1 and Threshold 2.
//    Log.i("10Comp frame", warp10.get(0,2)[0] + "x" + warp10.get(1,2)[0]);
//    Log.i("10Add frame", p.x + "x" + p.y);
//    Log.i("1 frame", warp.get(0,2)[0] + "x" + warp.get(1,2)[0]);

    cv::Scalar srg(255,255,0,0);//RGBA
    mMotionVectorMat = ComputeVector(cv::Point(warp10.at<float>(0,2),warp10.at<float>(1,2)),mMotionVectorMat,srg);

    cv::Scalar sr(255,0,0,0);//RGBA
    mMotionVectorMat = ComputeVector(p,mMotionVectorMat,sr);
    res = GOOD;
    if(mComputeVector_FinalMVector.x > mMax10FrameTranslationalMagnitude){
        res = TOO_HIGH;
    }

    cv::Scalar sg(0,255,0,0);
    mMotionVectorMat = ComputeVector(cv::Point(warp.at<float>(0,2),warp.at<float>(1,2)),mMotionVectorMat,sg);
    if(mComputeVector_FinalMVector.x > mMaxFrameTranslationalMagnitude){
        res = TOO_HIGH;
    }

    
    return res;
}
