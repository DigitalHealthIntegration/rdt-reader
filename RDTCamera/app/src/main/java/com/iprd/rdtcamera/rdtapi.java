package com.iprd.rdtcamera;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import static android.util.Config.LOGD;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_HIGH;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_LOW;
import static com.iprd.rdtcamera.AcceptanceStatus.GOOD;
import static com.iprd.rdtcamera.AcceptanceStatus.NOT_COMPUTED;
import static org.opencv.core.Core.BORDER_REFLECT101;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.meanStdDev;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class rdtapi {
    Config mConfig;
    AcceptanceStatus mAcceptanceStatus=new AcceptanceStatus();
    tensorFlow mTensorFlow=null;

    boolean computeBlur(Mat greyImage) {
        Mat laplacian=new Mat();
        Laplacian(greyImage, laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
        MatOfDouble median = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        meanStdDev(laplacian, median, std);
        // Release resources
        laplacian.release();
        double sharpness =(float) std.get(0,0)[0]*std.get(0,0)[0];
        Log.d("Sharpness","mSharpness " + sharpness);
        if (sharpness < mConfig.mMinSharpness){
            mAcceptanceStatus.mSharpness = TOO_LOW;
            return false;
        }
        mAcceptanceStatus.mSharpness = AcceptanceStatus.GOOD;
        return true;
    }

    boolean computeBrightness(Mat grey) {
        Scalar tempVal = mean(grey);
        double brightness = tempVal.val[0];
        Log.d("Brightness","mBrightness "+brightness);
        if (brightness > mConfig.mMaxBrightness) {
            mAcceptanceStatus.mBrightness = TOO_HIGH;
            return false;
        }else if (brightness < mConfig.mMinBrightness){
            mAcceptanceStatus.mBrightness = TOO_LOW;
            return false;
        }
        mAcceptanceStatus.mBrightness = AcceptanceStatus.GOOD;
        return true;
    }

    boolean computeDistortion(){
        if(mAcceptanceStatus.mBoundingBoxWidth > mConfig.mMaxScale){
            mAcceptanceStatus.mScale = TOO_HIGH;
            return false;
        }else if (mAcceptanceStatus.mBoundingBoxWidth < mConfig.mMinScale){
            mAcceptanceStatus.mScale = TOO_LOW;
            return false;
        }else mAcceptanceStatus.mScale = AcceptanceStatus.GOOD;

        if (mAcceptanceStatus.mBoundingBoxX > mConfig.mXMax){
            mAcceptanceStatus.mDisplacementX = TOO_HIGH;
            return false;
        }else if (mAcceptanceStatus.mBoundingBoxX < mConfig.mXMin){
            mAcceptanceStatus.mDisplacementX = TOO_LOW;
            return false;
        }else mAcceptanceStatus.mDisplacementX = GOOD;

        if (mAcceptanceStatus.mBoundingBoxY > mConfig.mYMax){
            mAcceptanceStatus.mDisplacementY = TOO_HIGH;
            return false;
        }else if (mAcceptanceStatus.mBoundingBoxY < mConfig.mYMin){
            mAcceptanceStatus.mDisplacementY = TOO_LOW;
            return false;
        }else mAcceptanceStatus.mDisplacementY = GOOD;
        mAcceptanceStatus.mPerspectiveDistortion= GOOD;
        return true;
    }


    public void init(Config c) {
        mConfig = c;
        mTensorFlow = new tensorFlow();
    }

    public AcceptanceStatus update(Bitmap capFrame) {
        Mat matinput = new Mat();
        Mat greyMat = new Mat();
        Utils.bitmapToMat(capFrame, matinput);
        Log.d("INPUT",capFrame.getWidth()+"x"+capFrame.getHeight());
        cvtColor(matinput, greyMat, Imgproc.COLOR_RGBA2GRAY);
//        {
//            mAcceptanceStatus.mBoundingBoxX = (short) (capFrame.getWidth()>>2);
//            mAcceptanceStatus.mBoundingBoxY = (short) (capFrame.getHeight()>>2);
//            mAcceptanceStatus.mBoundingBoxWidth = (short) (capFrame.getWidth()>>1);
//            mAcceptanceStatus.mBoundingBoxHeight = (short) (capFrame.getHeight()>>1);
//            mAcceptanceStatus.mRDTFound =  true;
//            if(true) return mAcceptanceStatus;
//        }
        mAcceptanceStatus.setDefaultStatus();

        Boolean [] rdtFound = new Boolean [] {new Boolean(false)};

        Rect roi = mTensorFlow.update(greyMat,rdtFound);
        mAcceptanceStatus.mRDTFound =  rdtFound[0].booleanValue();
        if(mAcceptanceStatus.mRDTFound) {

            roi.width = (roi.width - roi.x);
            roi.height = roi.height - roi.y ;

            roi.x = Math.max(0,roi.x);
            roi.y = Math.max(0,roi.y);
            roi.width = Math.min(greyMat.cols(),roi.width);
            roi.height = Math.min(greyMat.rows(),roi.height);

            mAcceptanceStatus.mBoundingBoxX = (short) roi.x;
            mAcceptanceStatus.mBoundingBoxY = (short) roi.y;
            mAcceptanceStatus.mBoundingBoxWidth = (short) (roi.width);
            mAcceptanceStatus.mBoundingBoxHeight = (short) (roi.height);
            mAcceptanceStatus.mRDTFound = rdtFound[0].booleanValue();
        }

        if(!rdtFound[0].booleanValue())return mAcceptanceStatus;
      //  if (!computeDistortion())return mAcceptanceStatus;
        Log.d("........ROI "," "+roi.x +"x"+roi.y + "x" +roi.width + "x" +roi.height);
        Log.d("greyMat "," " +greyMat.cols() + "x" +greyMat.height());

        Mat imageROI = greyMat.submat(roi);
        if(computeBlur(imageROI)){
            greyMat.release();
            return mAcceptanceStatus;
        }
        if(computeBrightness(imageROI)){
            greyMat.release();
            return mAcceptanceStatus;
        }
        greyMat.release();
        Log.d("FROMFUNCTION","Bounds "+ mAcceptanceStatus.mBoundingBoxX+"x"+mAcceptanceStatus.mBoundingBoxY+" Position "+mAcceptanceStatus.mBoundingBoxWidth+"x"+mAcceptanceStatus.mBoundingBoxHeight);
        return mAcceptanceStatus;
    }

    public  void setConfig(Config c) {
        mConfig = c;
    }

    public void term() {

    }
}