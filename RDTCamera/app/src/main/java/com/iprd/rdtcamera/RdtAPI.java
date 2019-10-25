package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.util.Log;

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
    private Config mConfig;
    private AcceptanceStatus mAcceptanceStatus;//=new AcceptanceStatus();
    private ObjectDetection mTensorFlow;//=null;
    private boolean mInprogress;// = false;
    private short mBrightness;
    private short mSharpness;
    private boolean mSaveNegativeData; //=false;

    public Config getmConfig() {
        return mConfig;
    }

    public AcceptanceStatus getmAcceptanceStatus() {
        return mAcceptanceStatus;
    }

    public ObjectDetection getmTensorFlow() {
        return mTensorFlow;
    }

    public boolean ismInprogress() {
        return mInprogress;
    }

    public short getmBrightness() {
        return mBrightness;
    }

    public short getmSharpness() {
        return mSharpness;
    }

    public boolean ismSaveNegativeData() {
        return mSaveNegativeData;
    }

    // private constructor , so that, we can only access via Builder
    private RdtAPI( RdtAPIBuilder rdtAPIBuilder){
        this.mConfig = rdtAPIBuilder.mConfig;
        this.mAcceptanceStatus = rdtAPIBuilder.mAcceptanceStatus;
        this.mTensorFlow = rdtAPIBuilder.mTensorFlow;
        this.mInprogress = rdtAPIBuilder.mInprogress;
        this.mBrightness = rdtAPIBuilder.mBrightness;
        this.mSharpness = rdtAPIBuilder.mSharpness;
        this.mSaveNegativeData = rdtAPIBuilder.mSaveNegativeData;
        this.mBrightness = rdtAPIBuilder.mBrightness;
        this.mSharpness = rdtAPIBuilder.mSharpness;

    }

    public static class RdtAPIBuilder {
        private Config mConfig;
        private AcceptanceStatus mAcceptanceStatus;//=new AcceptanceStatus();
        private ObjectDetection mTensorFlow;//=null;
        private boolean mInprogress;// = false;
        private short mBrightness;
        private short mSharpness;
        private boolean mSaveNegativeData; //=false;


        public RdtAPIBuilder(){
            this. mAcceptanceStatus=new AcceptanceStatus();
            this.mTensorFlow = null;
            this.mInprogress = false;
            this.mSaveNegativeData = false;

        }

        public RdtAPIBuilder mConfig(Config mConfig) {
            this.mConfig = mConfig;
            return this;
        }
        public RdtAPIBuilder  mAcceptanceStatus(AcceptanceStatus mAcceptanceStatus) {
            this.mAcceptanceStatus = mAcceptanceStatus;
            return this;
        }
        public RdtAPIBuilder mTensorFlow(ObjectDetection mTensorFlow) {
            this.mTensorFlow = mTensorFlow;
            return this;
        }
        public RdtAPIBuilder mInprogress(Boolean mInprogress) {
            this.mInprogress = mInprogress;
            return this;
        }
        public RdtAPIBuilder mBrightAndSharpness(Short mBrightness, Short mSharpness) {
            this.mBrightness = mBrightness;
            this.mSharpness = mSharpness;
            return this;
        }
        public RdtAPIBuilder mSaveNegativeData(boolean mSaveNegativeData) {
            this.mSaveNegativeData = mSaveNegativeData;
            return this;
        }

        public RdtAPI build() {
            RdtAPI rdtAPI1 =  new RdtAPI(this);
            //validateUserObject(rdtAPI1);
            return rdtAPI1;
        }
        public RdtAPIBuilder setSaveImages(boolean b){
            if(this.mTensorFlow!= null) {
                this.mTensorFlow.setSaveImages(b);
            }
            return this;
        }

        public RdtAPIBuilder setTopThreshold(double top){
            if(this.mTensorFlow!= null){
                this.mTensorFlow.setTopThreshold(top);
            }
            return this;
        }
        public RdtAPIBuilder setBottomThreshold(double bot){
            if(this.mTensorFlow!= null) {
                this.mTensorFlow.setBottomThreshold(bot);
            }
            return this;
        }

        public boolean isInProgress(){
            return mInprogress;
        }

        private boolean computeBlur(Mat greyImage,AcceptanceStatus ret) {
            Mat laplacian=new Mat();
            Laplacian(greyImage, laplacian, CV_16S, 3, 1, 0, BORDER_REFLECT101);
            MatOfDouble median = new MatOfDouble();
            MatOfDouble std = new MatOfDouble();
            meanStdDev(laplacian, median, std);
            // Release resources
            laplacian.release();
            double sharpness =(float) std.get(0,0)[0]*std.get(0,0)[0];
            mSharpness = (short) sharpness;
            //Log.d("Sharpness","mSharpness " + sharpness);
            if (sharpness < mConfig.mMinSharpness){
                ret.mSharpness = TOO_LOW;
                return false;
            }
            ret.mSharpness = AcceptanceStatus.GOOD;
            return true;
        }

        private boolean computeBrightness(Mat grey,AcceptanceStatus ret) {
            Scalar tempVal = mean(grey);
            double brightness = tempVal.val[0];
            mBrightness = (short) brightness;
            //Log.d("Brightness","mBrightness "+brightness);
            if (brightness > mConfig.mMaxBrightness) {
                ret.mBrightness = TOO_HIGH;
                return false;
            }else if (brightness < mConfig.mMinBrightness){
                ret.mBrightness = TOO_LOW;
                return false;
            }
            ret.mBrightness = AcceptanceStatus.GOOD;
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


        public RdtAPIBuilder init(Config c) {
            this.mConfig = c;
            this.mTensorFlow = new ObjectDetection(mConfig.mTfliteB);
            return this;
        }

        public AcceptanceStatus update(Bitmap capFrame) {
            mInprogress = true;
            mBrightness = -1;
            mSharpness = -1;
            Mat matinput = new Mat();
            Mat greyMat = new Mat();
            AcceptanceStatus ret= new AcceptanceStatus();

            try {
                Utils.bitmapToMat(capFrame, matinput);
                //Log.d("INPUT",capFrame.getWidth()+"x"+capFrame.getHeight());
                cvtColor(matinput, greyMat, Imgproc.COLOR_RGBA2GRAY);
                Boolean[] rdtFound = new Boolean[]{new Boolean(false)};
                Rect roi = mTensorFlow.update(greyMat, rdtFound);
                ret.mRDTFound = rdtFound[0].booleanValue();
                if (ret.mRDTFound) {
                    roi.width = (roi.width - roi.x);
                    roi.height = roi.height - roi.y;
                    roi.x = Math.max(0, roi.x);
                    roi.y = Math.max(0, roi.y);
                    roi.width = Math.max(0, roi.width);
                    roi.height = Math.max(0, roi.height);

                    roi.width = Math.min(greyMat.cols() - roi.x, roi.width);
                    roi.height = Math.min(greyMat.rows() - roi.y, roi.height);

                    ret.mBoundingBoxX = (short) roi.x;
                    ret.mBoundingBoxY = (short) roi.y;
                    ret.mBoundingBoxWidth = (short) (roi.width);
                    ret.mBoundingBoxHeight = (short) (roi.height);
                    ret.mRDTFound = rdtFound[0].booleanValue();
                }

                if (!rdtFound[0].booleanValue()) return ret;

                Mat imageROI = greyMat.submat(roi);
                if (!computeBlur(imageROI,ret)) {
                    return ret;
                }
                if (!computeBrightness(imageROI,ret)) {
                    return ret;
                }
            } catch (Exception e) {
            } finally {
                if(((!ret.mRDTFound) && mTensorFlow.getSaveImages())||(mSaveNegativeData&&ret.mRDTFound)){
                    saveImage(capFrame,"Color");
                }
                greyMat.release();
                matinput.release();
                mInprogress = false;
            }
            return ret;
        }

        public  void setConfig(Config c) {
            mConfig = c;
        }

        public void term() {

        }

        private void validateUserObject(RdtAPI rdtAPI) {
            //Do some basic validations to check
            //if user object does not break any assumption of system
        }

    }

}