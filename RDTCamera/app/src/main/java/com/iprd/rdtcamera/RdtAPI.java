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
import static com.iprd.rdtcamera.ImageRegistration.getTransformation;
import static com.iprd.rdtcamera.Utils.SaveMatrix;
import static com.iprd.rdtcamera.Utils.rotateRect;
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
import static org.opencv.imgproc.Imgproc.pyrDown;
import static org.opencv.imgproc.Imgproc.rectangle;

public class RdtAPI {
    private Config mConfig;
    private AcceptanceStatus mAcceptanceStatus;//=new AcceptanceStatus();
    private ObjectDetection mTensorFlow;//=null;
    private static boolean mInprogress = false;
    private static boolean mRDTProcessing = false;
    private short mBrightness;
    private short mSharpness;
    Mat mLocalcopy;
    Mat mRefPyr=null;
    boolean mPlaybackMode;
    long mTensorFlowProcessTime;
    long mPreProcessingTime;
    long mPostProcessingTime;

    public void setRotation(boolean mSetRotation) {
        this.mSetRotation = mSetRotation;
    }
    public void saveInput(boolean b) {
        this.mSaveInput = b;
    }
    boolean mSaveInput=false;
    boolean mSetRotation=false;

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

    public void setSavePoints(boolean b){
        mTensorFlow.setSavePoints(b);
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

    /*public void SetText(String message,AcceptanceStatus status){
        Scalar sr= new Scalar(255,0,0,0);
        Scalar sg= new Scalar(0,0,255,0);
        Scalar s;
        s = (status.mRDTFound)?sg:sr;
        putText(mLocalcopy,message,new Point((mLocalcopy.cols()>>2), mLocalcopy.rows()-50),0, 1,s,2,LINE_4,false);
    }*/
    public void SetText(String message, AcceptanceStatus status) {
        Scalar sr = new Scalar(255, 0, 0, 0);
        Scalar sg = new Scalar(0, 0, 255, 0);
        Scalar s;
        s = (status.mRDTFound) ? sg : sr;
        int yoff = 100;
        message.split("\n");
        for (String s1 : message.split("\n")) {
            putText(mLocalcopy, s1, new Point((mLocalcopy.cols() * 3) >> 2, yoff), 0, 3, s, 3, LINE_AA, false);
            yoff += 90;
        }
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

    private Mat FindMotion(Mat inp){
        Mat ref = new Mat();
        pyrDown(inp, ref);
        pyrDown(ref, ref);
        pyrDown(ref, ref);
        //pyrDown(ref, ref);
        Mat warpMatrix=null;
        if(mRefPyr!= null) {
            warpMatrix = getTransformation(mRefPyr, ref);
        }
        mRefPyr = ref.clone();
        return warpMatrix;
    }

    public AcceptanceStatus checkFrame(Bitmap capFrame) {
        mInprogress = true;
        mBrightness = -1;
        mSharpness = -1;
        Mat matinput = new Mat();
        Mat greyMat = new Mat();
        Mat greyMatResized = new Mat();
        Mat mPrevious;
        mPreProcessingTime = mTensorFlowProcessTime = mPostProcessingTime= 0;

        AcceptanceStatus ret= new AcceptanceStatus();
        try {
            long st  = System.currentTimeMillis();
            Utils.bitmapToMat(capFrame, matinput);
            cvtColor(matinput, greyMat, Imgproc.COLOR_RGBA2GRAY);
            Mat warpmat = FindMotion(greyMat);
            if(warpmat!=null) {
                Log.i("Tx-Ty",warpmat.get(0, 2)[0] +"x"+warpmat.get(1, 2)[0]);
                if (Math.abs(warpmat.get(0, 2)[0]) > mConfig.mMaxAllowedTranslationX || Math.abs(warpmat.get(1, 2)[0]) > mConfig.mMaxAllowedTranslationY) {
                    ret.mSteady = TOO_HIGH;
                    mPreProcessingTime = System.currentTimeMillis() - st;
                    Log.i("Motion Detected", "Too much Motion");
                    return ret;
                }else{
                    ret.mSteady = GOOD;
                    // if we have previous Bounding box then we should be able to translate it.
                    // update the new bounding box result.
                    int level = 1<<3;//3 level motion translates to 2**3 in level 0;
                    double tx= warpmat.get(0,2)[0]*level;
                    double ty= warpmat.get(1,2)[0]*level;
                    //Speculate point



                }
            }

            if(mSetRotation)greyMat=com.iprd.rdtcamera.Utils.rotateFrame(greyMat,-90);
            org.opencv.core.Size sz= new org.opencv.core.Size(1280, 720);
            Imgproc.resize(greyMat,greyMatResized,sz,0.0,0.0,INTER_CUBIC);
            mPreProcessingTime  = System.currentTimeMillis()-st;

            //process frame
            Rect detectedRoi=null;
            boolean rdtfound= false;
            if(!mRDTProcessing) {
                 detectedRoi = ProcessRDT(ret, matinput, greyMatResized);
            }

            if(mPlaybackMode) {
                mLocalcopy = matinput.clone();
            }
            if (!ret.mRDTFound) return ret;
            Mat imageROI = greyMatResized.submat(detectedRoi);
            if (!computeBlur(imageROI,ret)) {
                return ret;
            }
            if (!computeBrightness(imageROI,ret)) {
                return ret;
            }
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

    private Rect ProcessRDT(AcceptanceStatus retStatus,Mat inputmat,Mat reszgreymat){
        mRDTProcessing = true;
        Rect detectedRoi=null;
        try {
            Boolean[] rdtFound = new Boolean[]{new Boolean(false)};
            long updatetimest = System.currentTimeMillis();
            detectedRoi = mTensorFlow.update(reszgreymat, rdtFound);
            mTensorFlowProcessTime = System.currentTimeMillis() - updatetimest;
            mPostProcessingTime = System.currentTimeMillis();
            retStatus.mRDTFound = rdtFound[0].booleanValue();
            if (retStatus.mRDTFound) {
                Rect roi = detectedRoi.clone();
                if (mSetRotation) {
                    roi = rotateRect(reszgreymat, roi, -90);
                }
                float wfactor = 0;
                float hfactor = 0;
                if (mPlaybackMode) {
                    wfactor = inputmat.cols() / 1280.f;
                    hfactor = inputmat.rows() / 720f;
                } else {
                    wfactor = inputmat.cols() / 720.0f;
                    hfactor = inputmat.rows() / 1280f;
                }
                //handle rotation TBD
                if (mSaveInput || mPlaybackMode)
                    rectangle(inputmat, new Point(roi.x * wfactor, roi.y * hfactor), new Point((roi.x + roi.width) * wfactor, (roi.y + roi.height) * hfactor), new Scalar(255, 0, 0, 0), 4, LINE_AA, 0);
                if (mSaveInput) SaveMatrix(inputmat, "output");

                retStatus.mBoundingBoxX = (short) (roi.x * wfactor);
                retStatus.mBoundingBoxY = (short) (roi.y * hfactor);
                retStatus.mBoundingBoxWidth = (short) (roi.width * wfactor);
                retStatus.mBoundingBoxHeight = (short) (roi.height * hfactor);
            }
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
        finally {
            mRDTProcessing = false;
        }
        return detectedRoi;
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