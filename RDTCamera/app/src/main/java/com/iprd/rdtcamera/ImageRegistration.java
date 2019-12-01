package com.iprd.rdtcamera;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import static com.iprd.rdtcamera.CvUtils.PrintAffineMat;
import static com.iprd.rdtcamera.CvUtils.scaleAffineMat;
import static com.iprd.rdtcamera.Utils.SaveMatrix;
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
    static String TAG= InformationStatus.class.getName();
    static int REGISTRATION_LEVEL=4;
    static Mat mRefPyr=null;
    public static Mat GetTransform(Mat refM, Mat insM) {
        Mat ref = new Mat();
        pyrDown(refM, ref);
        Mat ins = new Mat();
        pyrDown(insM, ins);
        Mat warpMatrix = getTransformation(ref, ins);
        return warpMatrix;
    }

    public static Mat FindMotion(Mat inp,boolean saveref) {
        Mat ins = new Mat();
        pyrDown(inp, ins);
        for (int i = 0; i < REGISTRATION_LEVEL - 1; i++){
            pyrDown(ins, ins);
        }
        ins = DetectEdges(ins);
        Mat warpMatrix=null;
        if(mRefPyr!= null) {
            warpMatrix = getTransformation(mRefPyr,ins);
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
        if(warpMatrix != null) {
//            SaveMatrix(ref, "n1");
//            SaveMatrix(ins, "n2");
        }
        return warpMatrix;
    }

    public static Mat ComputeMotion(Mat greyMat) {
        Mat warp;
        Mat warpmat = ImageRegistration.FindMotion(greyMat, true);
        if (warpmat != null) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            PrintAffineMat("warp ", warp);
            //ComputeVector
            Log.i(TAG, "Tx-Ty Inp "+ warpmat.get(0, 2)[0] + "x" + warpmat.get(1, 2)[0]);
        }else{
            warp = Mat.eye(2,3,CV_32F);
            warp.put(0,0,1.0);
            warp.put(1,1,1.0);
            warp.put(0,2,greyMat.width());
            warp.put(1,2,greyMat.height());
        }
        return warp;
    }

    public static Mat getTransformation(Mat ref, Mat ins) {
       // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        final int warp_mode = MOTION_TRANSLATION;
        Mat warpMatrix = Mat.eye(2,3,CV_32F);
        try {
            int numIter = 50;
            double terminationEps = 1e-3;
            TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, numIter, terminationEps);
            double r= findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, new Mat());
            if(r == -1){
                Log.e(TAG,"Rc== -1");
                return null;
            }
        }catch(Exception e){
            Log.e(TAG,"Exception in FindTransformECC");
            return null;
        }
        return warpMatrix;
    }
    public static Mat DetectEdges(Mat grayMat){
        //Matrices to store gradient and absolute gradient respectively
        Mat grad_x = new Mat();
        Mat abs_grad_x = new Mat();

        Mat grad_y = new Mat();
        Mat abs_grad_y = new Mat();
        //Calculating gradient in horizontal direction
        Imgproc.Sobel(grayMat, grad_x, CvType.CV_16S, 1, 0, 3, 1, 0);

        //Calculating gradient in vertical direction
        Imgproc.Sobel(grayMat, grad_y, CvType.CV_16S, 0, 1, 3, 1, 0);

        //Calculating absolute value of gradients in both the direction
        Core.convertScaleAbs(grad_x, abs_grad_x);
        Core.convertScaleAbs(grad_y, abs_grad_y);

        //Calculating the resultant gradient
        Mat sobel = new Mat(); //Mat to store the final result
        Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 1, sobel);

        grad_x.release();
        abs_grad_x.release();
        grad_y.release();
        abs_grad_y.release();

        return sobel;
    }

}
