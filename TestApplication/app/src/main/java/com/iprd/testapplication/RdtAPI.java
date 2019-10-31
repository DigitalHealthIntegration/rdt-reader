package com.iprd.testapplication;

import android.graphics.Bitmap;

import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Mat;

import org.bytedeco.opencv.global.opencv_imgproc;
import org.opencv.core.Core;
import org.opencv.core.MatOfDouble;

import java.nio.MappedByteBuffer;

import static com.iprd.testapplication.AcceptanceStatus.GOOD;
import static com.iprd.testapplication.AcceptanceStatus.TOO_HIGH;
import static com.iprd.testapplication.AcceptanceStatus.TOO_LOW;
import static com.iprd.testapplication.Utils.saveImage;
import static org.bytedeco.opencv.global.opencv_core.BORDER_REFLECT101;
import static org.bytedeco.opencv.global.opencv_core.CV_16S;
import static org.bytedeco.opencv.global.opencv_core.mean;
import static org.bytedeco.opencv.global.opencv_core.meanStdDev;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_RGBA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_4;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.Laplacian;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;


public class RdtAPI {
    private Config mConfig;
    private AcceptanceStatus mAcceptanceStatus;//=new AcceptanceStatus();
    private ObjectDetection mTensorFlow;//=null;
    private boolean mInprogress;// = false;
    private short mBrightness;
    private short mSharpness;
    Mat mLocalcopy;

    public Bitmap getLocalcopyAsBitmap() {
        //Bitmap b  = Bitmap.createBitmap(mLocalcopy.cols(), mLocalcopy.rows(), Bitmap.Config.ARGB_8888);
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
        Bitmap b = converterToBitmap.convert(converterToMat.convert(mLocalcopy));
        return b;
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


    private boolean computeBlur(Mat greyImage,AcceptanceStatus ret) {
        Mat laplacian=new Mat();
        Laplacian(greyImage, laplacian, CV_16S, 3, 1.0, 0.0, BORDER_REFLECT101);
//        MatOfDouble median = new MatOfDouble();
//        MatOfDouble std = new MatOfDouble();
        Mat median=new Mat(),std=new Mat();
        opencv_core.meanStdDev(laplacian, median, std);
        // Release resources
        laplacian.release();
        //double sharpness =(float) std.get(0,0)[0]*std.get(0,0)[0];
        double s=std.createIndexer().getDouble();
        double sharpness = s*s;
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
        double brightness = tempVal.get();
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
        Mat matinput = new Mat();
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
        Frame f = converterToBitmap.convert(capFrame);
        matinput = converterToMat.convert(f);
        //bitmapToMat(capFrame, matinput);
        return processMatrix(capFrame, matinput);
    }

    public void SetText(String message,AcceptanceStatus status){
        Scalar sr= new Scalar(255,0,0,0);
        Scalar sg= new Scalar(0,255,0,0);
        Scalar s;
        s = (status.mRDTFound)?sg:sr;
        putText(mLocalcopy,message,new Point((mLocalcopy.cols()>>2)*3, 50),0, 1,s,2,LINE_AA,false);
    }

    public AcceptanceStatus processMatrix(Bitmap capFrame, Mat matinput) {
        mInprogress = true;
        mBrightness = -1;
        mSharpness = -1;

        Mat greyMat = new Mat();
        AcceptanceStatus ret= new AcceptanceStatus();

        try {
            //Log.d("INPUT",capFrame.getWidth()+"x"+capFrame.getHeight());
            cvtColor(matinput, greyMat, COLOR_RGBA2GRAY);
            Boolean[] rdtFound = new Boolean[]{new Boolean(false)};
            Rect roi = mTensorFlow.update(greyMat, rdtFound);
            ret.mRDTFound = rdtFound[0].booleanValue();
            if (ret.mRDTFound) {
                roi.width(roi.width() - roi.x());
                roi.height(roi.height() - roi.y());
                roi.x(Math.max(0, roi.x()));
                roi.y(Math.max(0, roi.y()));
                roi.width(Math.max(0, roi.width()));
                roi.height(Math.max(0, roi.height()));

                roi.width(Math.min(greyMat.cols() - roi.x(), roi.width()));
                roi.height(Math.min(greyMat.rows() - roi.y(), roi.height()));

                ret.mBoundingBoxX = (short) roi.x();
                ret.mBoundingBoxY = (short) roi.y();
                ret.mBoundingBoxWidth = (short) (roi.width());
                ret.mBoundingBoxHeight = (short) (roi.height());
                ret.mRDTFound = rdtFound[0].booleanValue();
                opencv_imgproc.rectangle(matinput, new Point(roi.x(), roi.y()), new Point(roi.x()+roi.width(), roi.y()+roi.height()), new Scalar(255,0, 0,0),4,LINE_AA,0);
            }
            mLocalcopy = matinput;

            if (!rdtFound[0].booleanValue()) return ret;

            Mat imageROI = greyMat.apply(roi);

            //Mat imageROI = greyMat.submat(roi);
            if (!computeBlur(imageROI,ret)) {
                return ret;
            }
            if (!computeBrightness(imageROI,ret)) {
                return ret;
            }
        } catch (Exception e) {
        } finally {
            if(((!ret.mRDTFound) && mTensorFlow.getSaveImages())||(mSaveNegativeData&&ret.mRDTFound)){
                if(capFrame != null) saveImage(capFrame,"Color");
            }
            greyMat.release();
            //matinput.release();
            mInprogress = false;
        }
        return ret;
    }


    // private constructor , so that, we can only access via Builder
    private RdtAPI( RdtAPIBuilder rdtAPIBuilder){
        this.mConfig = rdtAPIBuilder.mConfig;
        this.mTensorFlow = new ObjectDetection(this.mConfig.mMappedByteBuffer);
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

//        public RdtAPIBuilder setByteModel(byte[] mTfliteB) {
//            mConfig.setmTfliteB(mTfliteB);
//            return this;
//        }

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