package com.iprd.rdtcamera;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import static com.iprd.rdtcamera.CvUtils.PrintAffineMat;
import static com.iprd.rdtcamera.CvUtils.scaleAffineMat;
import static com.iprd.rdtcamera.Utils.SaveMatrix;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
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

    static int REGISTRATION_LEVEL=3;
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
            warpMatrix = getTransformation(mRefPyr,ins);
        }
        if(saveref)mRefPyr = ins.clone();
//        SaveMatrix(mRefPyr,"m1");
//        SaveMatrix(ins,"m2");
        return warpMatrix;
    }

    public static Mat FindMotionRefIns(Mat inp,Mat refe,boolean resize){
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
        Mat warpmat=null,warp=null;
        warpmat = getTransformation(ref, ins);
        if (warpmat != null) {
            warp = scaleAffineMat(warpmat, REGISTRATION_LEVEL);
            PrintAffineMat("warpRI", warp);
            //ComputeVector
            Log.i("Tx-Ty 10 Inp", warpmat.get(0, 2)[0] + "x" + warpmat.get(1, 2)[0]);
        }else{
            warp = Mat.eye(2,3,CV_32F);
            warp.put(0,0,1.0);
            warp.put(1,1,1.0);
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

    public static Mat getTransformation(Mat ref, Mat ins) {
       // Log.d("Transform",ref.cols()+"x"+ref.rows()+ " " +ins.cols()+"x"+ins.rows());
        final int warp_mode = MOTION_TRANSLATION;
        Mat warpMatrix = Mat.eye(2,3,CV_32F);
        try {
            int numIter = 500;
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
