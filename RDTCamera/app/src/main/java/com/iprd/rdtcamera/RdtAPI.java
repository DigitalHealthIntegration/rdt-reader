package com.iprd.rdtcamera;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import static com.iprd.rdtcamera.AcceptanceStatus.GOOD;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_HIGH;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_LOW;
import static com.iprd.rdtcamera.Utils.saveImage;
import static org.opencv.core.Core.BORDER_REFLECT101;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.meanStdDev;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class RdtAPI {
    Config mConfig;
    AcceptanceStatus mAcceptanceStatus=new AcceptanceStatus();
    ObjectDetection mTensorFlow=null;
    boolean mInprogress = false;
    short mBrightness,mSharpness;
    boolean mSaveNegativeData=false;

    public void setSaveImages(boolean b){
        if(mTensorFlow!= null)mTensorFlow.setSaveImages(b);
    }

    public void setTopThreshold(double top){
        if(mTensorFlow!= null)mTensorFlow.setTopThreshold(top);
    }
    public void setBottomThreshold(double bot){
        if(mTensorFlow!= null)mTensorFlow.setBottomThreshold(bot);
    }


    public boolean isInProgress(){
        return mInprogress;
    }

    private boolean computeBlur(Mat greyImage) {
        Mat laplacian=new Mat();
        Laplacian(greyImage, laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
        MatOfDouble median = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        meanStdDev(laplacian, median, std);
        // Release resources
        laplacian.release();
        double sharpness =(float) std.get(0,0)[0]*std.get(0,0)[0];
        mSharpness = (short) sharpness;
       // Log.d("Sharpness","mSharpness " + sharpness);
        if (sharpness < mConfig.mMinSharpness){
            mAcceptanceStatus.mSharpness = TOO_LOW;
            return false;
        }
        mAcceptanceStatus.mSharpness = AcceptanceStatus.GOOD;
        return true;
    }

    private boolean computeBrightness(Mat grey) {
        Scalar tempVal = mean(grey);
        double brightness = tempVal.val[0];
        mBrightness = (short) brightness;
        //Log.d("Brightness","mBrightness "+brightness);
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

    private boolean computeDistortion(){
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
        mTensorFlow = new ObjectDetection(mConfig.mTfliteB);
    }

    public AcceptanceStatus update(Bitmap capFrame) {
        mInprogress = true;
        mBrightness = -1;
        mSharpness = -1;
        Mat matinput = new Mat();
        Mat greyMat = new Mat();
        try {
            Utils.bitmapToMat(capFrame, matinput);
            //Log.d("INPUT",capFrame.getWidth()+"x"+capFrame.getHeight());
            cvtColor(matinput, greyMat, Imgproc.COLOR_RGBA2GRAY);
            mAcceptanceStatus.setDefaultStatus();
            Boolean[] rdtFound = new Boolean[]{new Boolean(false)};

            Rect roi = mTensorFlow.update(greyMat, rdtFound);
            mAcceptanceStatus.mRDTFound = rdtFound[0].booleanValue();
            if (mAcceptanceStatus.mRDTFound) {
                roi.width = (roi.width - roi.x);
                roi.height = roi.height - roi.y;
                roi.x = Math.max(0, roi.x);
                roi.y = Math.max(0, roi.y);
                roi.width = Math.max(0, roi.width);
                roi.height = Math.max(0, roi.height);

                roi.width = Math.min(greyMat.cols() - roi.x, roi.width);
                roi.height = Math.min(greyMat.rows() - roi.y, roi.height);

                mAcceptanceStatus.mBoundingBoxX = (short) roi.x;
                mAcceptanceStatus.mBoundingBoxY = (short) roi.y;
                mAcceptanceStatus.mBoundingBoxWidth = (short) (roi.width);
                mAcceptanceStatus.mBoundingBoxHeight = (short) (roi.height);
                mAcceptanceStatus.mRDTFound = rdtFound[0].booleanValue();
            }

            if (!rdtFound[0].booleanValue()) return mAcceptanceStatus;
            //  if (!computeDistortion())return mAcceptanceStatus;
            //Log.d("........ROI "," "+roi.x +"x"+roi.y + "x" +roi.width + "x" +roi.height);

            Mat imageROI = greyMat.submat(roi);
            // mTensorFlow.SaveROIImage(imageROI,0,0,imageROI.width(),imageROI.height());

            if (!computeBlur(imageROI)) {
                return mAcceptanceStatus;
            }
            if (!computeBrightness(imageROI)) {
                return mAcceptanceStatus;
            }
        } catch (Exception e) {
        } finally {
            if(((!mAcceptanceStatus.mRDTFound) && mTensorFlow.getSaveImages())||(mSaveNegativeData&&mAcceptanceStatus.mRDTFound)){
                saveImage(capFrame,"Color");
            }
            greyMat.release();
            matinput.release();
            mInprogress = false;
        }
        return mAcceptanceStatus;
   }

    public  void setConfig(Config c) {
        mConfig = c;
    }

    public void term() {

    }
}