package com.iprd.rdtcamera;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.WARP_INVERSE_MAP;
import static org.opencv.imgproc.Imgproc.pyrDown;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.imgproc.Imgproc.warpPerspective;
import static org.opencv.video.Video.MOTION_AFFINE;
import static org.opencv.video.Video.MOTION_EUCLIDEAN;
import static org.opencv.video.Video.MOTION_HOMOGRAPHY;
import static org.opencv.video.Video.MOTION_TRANSLATION;
import static org.opencv.video.Video.findTransformECC;

public class ImageRegistration {

    static Mat mRefPyr=null;
    public static Mat GetTransform(Mat refM, Mat insM) {
        Mat ref = new Mat();
        pyrDown(refM, ref);
        Mat ins = new Mat();
        pyrDown(insM, ins);
        Mat warpMatrix = getTransformation(ref, ins);
        return warpMatrix;
    }

    public static Mat FindMotion(Mat inp,boolean saveref){
        Mat ins = new Mat();
        pyrDown(inp, ins);
        pyrDown(ins, ins);
        pyrDown(ins, ins);
        pyrDown(ins, ins);
        Mat warpMatrix=null;
        if(mRefPyr!= null) {
            warpMatrix = getTransformation(mRefPyr, ins);
        }
        if(saveref)mRefPyr = ins.clone();
        return warpMatrix;
    }

    public static Mat FindMotionRefIns(Mat inp,Mat refe){
        Mat ins = new Mat();
        pyrDown(inp, ins);
        pyrDown(ins, ins);
        pyrDown(ins, ins);
        pyrDown(ins, ins);
        Mat ref = new Mat();
        pyrDown(refe, ref);
        pyrDown(ref, ref);
        pyrDown(ref, ref);
        pyrDown(ref, ref);
        Mat warpMatrix=null;
        warpMatrix = getTransformation(ref, ins);
        return warpMatrix;
    }



    public static Mat getTransformation(Mat ref, Mat ins) {
       // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        final int warp_mode = MOTION_EUCLIDEAN;
        Mat warpMatrix = Mat.eye(2,3,CV_32F);
       try {
            int numIter = 50;
            double terminationEps = 1e-3;
            TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, numIter, terminationEps);
            findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, ins);
        }catch(Exception e){
            Log.e("Exception","Exception in FindTransformECC");
            return null;
        }
        return warpMatrix;
    }
}
