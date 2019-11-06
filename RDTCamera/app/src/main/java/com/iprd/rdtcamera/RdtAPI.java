package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.MappedByteBuffer;

import static com.iprd.rdtcamera.AcceptanceStatus.GOOD;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_HIGH;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_LOW;
import static com.iprd.rdtcamera.Utils.saveImage;
import static org.opencv.core.Core.BORDER_REFLECT101;
import static org.opencv.core.Core.LINE_4;
import static org.opencv.core.Core.LINE_AA;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.meanStdDev;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.INTER_LANCZOS4;
import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;

public class RdtAPI {
    private Config mConfig;
    private AcceptanceStatus mAcceptanceStatus;//=new AcceptanceStatus();
    private ObjectDetection mTensorFlow;//=null;
    private boolean mInprogress;// = false;
    private short mBrightness;
    private short mSharpness;
    Mat mLocalcopy;
    boolean mPlaybackMode;
    long mTensorFlowProcessTime;
    long mPreProcessingTime;
    long mPostProcessingTime;

    public long getPostProcessingTime() {
        return mPostProcessingTime;
    }
    public long getPreProcessingTime() {
        return mPreProcessingTime;
    }
    public long getTensorFlowProcessTime() {
        return mTensorFlowProcessTime;
    }

    public boolean isPlaybackMode() {
        return mPlaybackMode;
    }

    public void setmPlaybackMode(boolean mPlaybackMode) {
        this.mPlaybackMode = mPlaybackMode;
    }


    public void setSaveNegativeData(boolean mSaveNegativeData) {
        this.mSaveNegativeData = mSaveNegativeData;
    }

    private boolean mSaveNegativeData; //=false;

    public Config getConfig() {
        return mConfig;
    }

    public void setSaveImages(boolean b){
        mTensorFlow.setSaveImages(b);
    }
    public AcceptanceStatus getAcceptanceStatus() {
        return mAcceptanceStatus;
    }

    public ObjectDetection getTensorFlow() {
        return mTensorFlow;
    }

    public boolean isInprogress() {
        return mInprogress;
    }

    public short getBrightness() {
        return mBrightness;
    }

    public short getSharpness() {
        return mSharpness;
    }

    public boolean isSaveNegativeData() {
        return mSaveNegativeData;
    }

    public Bitmap getLocalcopyAsBitmap() {
        Bitmap b = Bitmap.createBitmap(mLocalcopy.cols(), mLocalcopy.rows(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(mLocalcopy, b);
        return b;
    }

    public void SetText(String message,AcceptanceStatus status){
        Scalar sr= new Scalar(255,0,0,0);
        Scalar sg= new Scalar(0,0,255,0);
        Scalar s;
        s = (status.mRDTFound)?sg:sr;
        putText(mLocalcopy,message,new Point((mLocalcopy.cols()>>2), mLocalcopy.rows()-50),0, 1,s,2,LINE_4,false);
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


    public AcceptanceStatus checkFrame(Bitmap capFrame) {
        mInprogress = true;
        mBrightness = -1;
        mSharpness = -1;
        Mat matinput = new Mat();
        Mat greyMat = new Mat();
        Mat greyMatResized = new Mat();

        AcceptanceStatus ret= new AcceptanceStatus();

        try {
            long st  = System.currentTimeMillis();
            Utils.bitmapToMat(capFrame, matinput);
            cvtColor(matinput, greyMat, Imgproc.COLOR_RGBA2GRAY);
            greyMat=com.iprd.rdtcamera.Utils.rotateFrame(greyMat,-90);

            org.opencv.core.Size sz= new org.opencv.core.Size(1280, 720);
            Log.d("IMAGESIZE:","WIDTH "+greyMat.width()+"HEIGHT "+greyMat.height());

            Imgproc.resize(greyMat,greyMatResized,sz,0.0,0.0,INTER_CUBIC);
            mPreProcessingTime  = System.currentTimeMillis()-st;

            Boolean[] rdtFound = new Boolean[]{new Boolean(false)};
            mTensorFlowProcessTime = System.currentTimeMillis();
            Rect roi = mTensorFlow.update(greyMatResized, rdtFound);
            mTensorFlowProcessTime =  System.currentTimeMillis()-mTensorFlowProcessTime;
            mPostProcessingTime  = System.currentTimeMillis();
            ret.mRDTFound = rdtFound[0].booleanValue();
            if (ret.mRDTFound) {

                ret.mBoundingBoxY = (short) ((1280-roi.x)/1280.0f*greyMat.width());
                ret.mBoundingBoxX = (short) (roi.y/720.0f*greyMat.height());
                ret.mBoundingBoxHeight = (short) (roi.width/1280.0f*greyMat.width());
                ret.mBoundingBoxWidth= (short) (roi.height/720.0f*greyMat.height());
                ret.mRDTFound = rdtFound[0].booleanValue();
                if(mPlaybackMode) rectangle(matinput, new Point(roi.x, roi.y), new Point(roi.x+roi.width, roi.y+roi.height), new Scalar(255,0, 0,0),4,LINE_AA,0);
            }
            if(mPlaybackMode) {
                mLocalcopy = matinput.clone();
            }

            if (!rdtFound[0].booleanValue()) return ret;

//            Mat imageROI = greyMat.submat(roi);
//            if (!computeBlur(imageROI,ret)) {
//                return ret;
//            }
//            if (!computeBrightness(imageROI,ret)) {
//                return ret;
//            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if(((!ret.mRDTFound) && mTensorFlow.getSaveImages())||(mSaveNegativeData&&ret.mRDTFound)){
//                saveImage(capFrame,"Color");
            }
            greyMat.release();
            matinput.release();
            mInprogress = false;
            mPostProcessingTime = System.currentTimeMillis()-mPostProcessingTime;
        }
        return ret;
    }


    // private constructor , so that, we can only access via Builder
    private RdtAPI( RdtAPIBuilder rdtAPIBuilder){
        mPlaybackMode=false;
        this.mConfig = rdtAPIBuilder.mConfig;
        if(this.mConfig.mMappedByteBuffer != null)this.mTensorFlow = new ObjectDetection(this.mConfig.mMappedByteBuffer);
        if(this.mConfig.mTfliteB != null)this.mTensorFlow = new ObjectDetection(this.mConfig.mTfliteB);
    }

    public static class RdtAPIBuilder {
        private Config mConfig;

        public  RdtAPIBuilder(){
            mConfig = new Config();
        }

        public RdtAPIBuilder setModel(MappedByteBuffer model){
            mConfig.setmMappedByteBuffer(model);
            return this;
        }

//        public RdtAPIBuilder mConfig(Config mConfig) {
//            this.mConfig = mConfig;
//            return this;
//        }

        public RdtAPI build() {
            RdtAPI rdtAPI =  new RdtAPI(this);
            //validateUserObject(rdtAPI1);
            return rdtAPI;
        }

        public RdtAPIBuilder setByteModel(byte[] mTfliteB) {
            mConfig.setmTfliteB(mTfliteB);
            return this;
        }

        public RdtAPIBuilder setMinBrightness(float mMinBrightness) {
            mConfig.setmMinBrightness(mMinBrightness);
            return this;
        }

        public RdtAPIBuilder setMaxBrightness(float mMaxBrightness) {
            mConfig.setmMaxBrightness(mMaxBrightness);
            return this;
        }

        public RdtAPIBuilder setMinSharpness(float mMinSharpness) {
            mConfig.setmMinSharpness(mMinSharpness);
            return this;
        }

        public RdtAPIBuilder setYMax(short mYMax) {
            mConfig.setmYMax(mYMax);
            return this;
        }

        public RdtAPIBuilder setYMin(short mYMin) {
            mConfig.setmYMin(mYMin);
            return this;
        }

        public RdtAPIBuilder setXMax(short mXMax) {
            mConfig.setmXMax(mXMax);
            return this;
        }

        public RdtAPIBuilder setXMin(short mXMin) {
            mConfig.setmXMin(mXMin);
            return this;
        }

        public RdtAPIBuilder setMinScale(short mMinScale) {
            mConfig.setmMinScale(mMinScale);
            return this;
        }

        public RdtAPIBuilder setMaxScale(short mMaxScale) {
            mConfig.setmMaxScale(mMaxScale);
            return this;
        }

        private void validateUserObject(RdtAPI rdtAPI) {
            //Do some basic validations to check
            //if user object does not break any assumption of system
        }

    }

}