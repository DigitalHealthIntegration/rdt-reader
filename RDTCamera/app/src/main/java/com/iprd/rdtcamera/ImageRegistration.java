package com.iprd.rdtcamera;

import android.nfc.Tag;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import static com.iprd.rdtcamera.CvUtils.PrintAffineMat;
import static com.iprd.rdtcamera.CvUtils.scaleAffineMat;
import static com.iprd.rdtcamera.Utils.SaveMatrix;
import static org.opencv.core.Core.BORDER_DEFAULT;
import static org.opencv.core.Core.convertScaleAbs;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.WARP_INVERSE_MAP;
import static org.opencv.imgproc.Imgproc.pyrDown;
import static org.opencv.imgproc.Imgproc.pyrUp;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.imgproc.Imgproc.warpPerspective;
import static org.opencv.video.Video.MOTION_AFFINE;
import static org.opencv.video.Video.MOTION_EUCLIDEAN;
import static org.opencv.video.Video.MOTION_HOMOGRAPHY;
import static org.opencv.video.Video.MOTION_TRANSLATION;
import static org.opencv.video.Video.findTransformECC;

public class ImageRegistration {

    static int REGISTRATION_LEVEL=5;
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

        for(int i=0;i<REGISTRATION_LEVEL-1;i++){
            pyrDown(ins, ins);
        }
        Mat warpMatrix=null;
        if(mRefPyr!= null) {
            warpMatrix = getTransformation(ins,mRefPyr);
        }
        if(saveref)mRefPyr = ins.clone();
//        SaveMatrix(mRefPyr,"m1");
//        SaveMatrix(ins,"m2");
        return warpMatrix;
    }

    public static Mat FindMotionRefIns(Mat refe,Mat inp,Mat warpmat ,boolean resize){
        Mat ins = new Mat();
        Mat ref = new Mat();
        if(resize){
            Size s = new Size(inp.width()>>REGISTRATION_LEVEL,inp.height()>>REGISTRATION_LEVEL);
            ins = new Mat((int)s.width,(int)s.height,inp.type());
            ref = new Mat((int)s.width,(int)s.height,inp.type());
            Imgproc.resize(inp,ins,s, 0.0, 0.0, INTER_CUBIC);
            Imgproc.resize(refe,ref,s, 0.0, 0.0, INTER_CUBIC);
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
        Mat warp=null;
        double ret  = updateTransformationMat(DetectEdges(ins),DetectEdges(ref),warpmat);
        if (ret >0.0) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            PrintAffineMat("warpRI", warp);
            //ComputeVector
            Log.i("Tx-Ty 10 Inp", warpmat.get(0, 2)[0] + "x" + warpmat.get(1, 2)[0]);
        }else{
            warp = Mat.eye(2,3,CV_32F);
            warpmat=warp.clone();
            warp.put(0,2,refe.width());
            warp.put(1,2,refe.height());
        }
        ref.release();
        ins.release();
        return warp;
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
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        //Calculating the resultant gradient
        Mat sobel = new Mat(); //Mat to store the final result
        Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 1, sobel);

        grad_x.release();
        abs_grad_x.release();
        grad_y.release();
        abs_grad_y.release();

        return sobel;
    }

    static Mat LaplacianCompute(Mat inp){
        Mat ins1 = new Mat();
        pyrDown(inp, ins1);
        Mat up=new Mat(inp.width(),inp.height(),inp.type());
        pyrUp(ins1,up);

        Mat lap=new Mat(inp.width(),inp.height(),CvType.CV_16S);
        //Core.addWeighted(inp,1.0,up,-1.0,255,lap,lap.type());
        //convertScaleAbs(lap,up);
        Core.subtract(inp,up,lap);
        convertScaleAbs(lap, up);
//        SaveMatrix(up,"insup");
        return up;
    }
    public static Mat FindMotionLaplacianRefIns(Mat refe,Mat inp,Mat warpmat ,boolean resize){
        Mat ins = new Mat();
        Mat ref = new Mat();
        Mat ins1 = new Mat();

        int w=0,h=0;
        if(resize){
            Size s = new Size(inp.width()>>REGISTRATION_LEVEL,inp.height()>>REGISTRATION_LEVEL);
            ins = new Mat((int)s.width,(int)s.height,inp.type());
            ref = new Mat((int)s.width,(int)s.height,inp.type());
            Imgproc.resize(inp,ins,s, 0.0, 0.0, INTER_CUBIC);
            Imgproc.resize(refe,ref,s, 0.0, 0.0, INTER_CUBIC);
        }else {
            pyrDown(inp, ins);
            w = ins.rows();
            h = ins.cols();
            for (int i = 0; i < REGISTRATION_LEVEL-1 ; i++) {
                w = ins.rows();
                h = ins.cols();
                pyrDown(ins, ins);
            }

//            ins = LaplacianCompute(ins);
//
            int kernel_size = 3;
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            Mat temp = new Mat();
//            Laplacian( ins, temp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
//            convertScaleAbs( temp, ins,1,0);

            pyrDown(ins, ins1);
            Mat up=new Mat(w,h,ins.type());
            pyrUp(ins1,up);
//            //SaveMatrix(up,"insup");
            Core.subtract(ins,up,ins);
//            //ins = ins - up;

            pyrDown(refe, ref);
            w = ref.rows();
            h = ref.cols();
            for (int i = 0; i < REGISTRATION_LEVEL-1 ; i++) {
                w = ref.rows();
                h = ref.cols();
                pyrDown(ref, ref);
            }
//            ref = LaplacianCompute(ref);

//            Laplacian( ref, temp, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
//            convertScaleAbs( temp, ref,1,0);

            pyrDown(ref, ins1);
            up=new Mat(w,h,ref.type());
            pyrUp(ins1,up);
            //SaveMatrix(up,"refup");
            Core.subtract(ref,up,ref);
//            Mat dst = new Mat();
//            Core.convertScaleAbs(ins,dst,1,128);
//            SaveMatrix(dst,"ref");

        }
        Mat warp=null;
        double ret  = updateTransformationMat(ins,ref,warpmat);
        if (ret >0.0) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            PrintAffineMat("warpRI", warp);
            //ComputeVector
            Log.i("Tx-Ty 10 Inp", warpmat.get(0, 2)[0] + "x" + warpmat.get(1, 2)[0]);
        }else{
            warp = Mat.eye(2,3,CV_32F);
            warpmat=warp.clone();
            warp.put(0,2,refe.width());
            warp.put(1,2,refe.height());
        }
        ref.release();
        ins.release();
        return warp;
    }

    public static Mat ComputeMotion(Mat greyMat) {
        Mat warp;
        Mat warpmat = ImageRegistration.FindMotion(greyMat, true);
        if (warpmat != null) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            PrintAffineMat("warp", warp);
            //ComputeVector
            Log.i("Tx-Ty Inp", warpmat.get(0, 2)[0] + "x" + warpmat.get(1, 2)[0]);
        }else{
            warp = Mat.eye(2,3,CV_32F);
            warp.put(0,0,1.0);
            warp.put(1,1,1.0);
            warp.put(0,2,greyMat.width());
            warp.put(1,2,greyMat.height());
        }
        return warp;
    }
    public static double updateTransformationMat(Mat ref, Mat ins,Mat warpMatrix) {
        //SaveMatrix(ref,"ins");
        //SaveMatrix(ins,"ref");
        // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        final int warp_mode = MOTION_TRANSLATION;
        double ret = -1.0;
        try {
            int numIter = 50;
            double terminationEps = 1e-3;
            TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, numIter, terminationEps);
            ret = findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, new Mat());
            Log.i("findTransformECC", " "+String.valueOf(ret));
        }catch(Exception e){
            Log.e("Exception","Exception in FindTransformECC");
            return -1.0;
        }
        return ret;
    }
    public static Mat getTransformation(Mat ref, Mat ins) {
       // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        final int warp_mode = MOTION_TRANSLATION;
        Mat warpMatrix = Mat.eye(2,3,CV_32F);
        try {
            int numIter = 50;
            double terminationEps = 1e-3;
            TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, numIter, terminationEps);
            findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, new Mat());
        }catch(Exception e){
            Log.e("Exception","Exception in FindTransformECC");
            return null;
        }
        return warpMatrix;
    }
}
