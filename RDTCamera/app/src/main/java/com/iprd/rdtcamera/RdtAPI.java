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
import static com.iprd.rdtcamera.ObjectDetection.warpPoint;
import static com.iprd.rdtcamera.Utils.SaveMatrix;
import static com.iprd.rdtcamera.Utils.rotateRect;
import static com.iprd.rdtcamera.Utils.saveImage;
import static java.time.Instant.MIN;
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
    private static volatile boolean mInprogress = false;

    private static volatile boolean mRDTProcessing = false;
    private static volatile boolean mRDTProcessingResultAvailable = false;
    Mat mInputMat;
    Mat mGreyMat;
    Mat mGreyMatResized;
    AcceptanceStatus mStatus=null;
    int mTaskID=-1;

    private short mBrightness;
    private short mSharpness;
    Mat mLocalcopy;
    Mat mRefPyr=null;
    boolean mPlaybackMode;
    long mTensorFlowProcessTime;
    long mPreProcessingTime;
    long mPostProcessingTime;
    Point mPreviousLTPoint;
    Point mPreviousRBPoint;
    boolean mPreviousStudy=false;
    boolean ismPreviousRDT=false;
    Mat mPreviousMat=null;
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
            Mat warpmat = ImageRegistration.FindMotion(greyMat,true);
            Point lt = null,rb=null;
            if(warpmat!=null) {
                //Log.i("Tx-Ty",warpmat.get(0, 2)[0] +"x"+warpmat.get(1, 2)[0]);
                if (Math.abs(warpmat.get(0, 2)[0]) > mConfig.mMaxAllowedTranslationX || Math.abs(warpmat.get(1, 2)[0]) > mConfig.mMaxAllowedTranslationY) {
                    ret.mSteady = TOO_HIGH;
                    mPreviousStudy = false;
                    mPreProcessingTime = System.currentTimeMillis() - st;
                    Log.i("Motion Detected", "Too much Motion");
                    return ret;
                }else{
                    if((mPreviousStudy)&&(ismPreviousRDT)) {
                        //if we have previous Bounding box then we should be able to translate it.
                        //update the new bounding box result.
                        //Speculate point
                        //Lets predict the next rectangle
                        //Tracking BB
                        ret.mSteady = GOOD;
                        if((null != mPreviousLTPoint)&&(null != mPreviousRBPoint) )
                        lt = warpPoint(new Point(mPreviousLTPoint.x/16.0,mPreviousLTPoint.y/16.0),warpmat);
                        lt.x = lt.x*16;
                        lt.y = lt.y*16;
                        //Log.i("Previous Mat", mPreviousLTPoint.x+"x"+mPreviousLTPoint.y+" "+mPreviousRBPoint.x+"x"+mPreviousRBPoint.y);
                        ClampBoundingBox(greyMat, lt);
                        ret.mBoundingBoxX = (short)(lt.x);
                        ret.mBoundingBoxY = (short)(lt.y);
                        ret.mBoundingBoxWidth = (short)((mPreviousRBPoint.x + ret.mBoundingBoxX)> greyMat.cols()? greyMat.cols() - ret.mBoundingBoxX:mPreviousRBPoint.x);////(short)((rb.x-lt.x)*16);
                        ret.mBoundingBoxHeight = (short)((mPreviousRBPoint.y + ret.mBoundingBoxY)> greyMat.rows()? greyMat.rows() - ret.mBoundingBoxY:mPreviousRBPoint.y);
                        //Log.i("Computed Mat", ret.mBoundingBoxX+"x"+ret.mBoundingBoxY+" "+ ret.mBoundingBoxWidth+"x"+ret.mBoundingBoxHeight);
                        ret.mRDTFound = true;
                    }else{
                        mPreviousStudy=true;
                    }
                }
            }
            ismPreviousRDT = false;
            //process frame
            Rect detectedRoi = null;
            if(true) {
                Mat rotatedmat = new Mat();
                if (mSetRotation) rotatedmat = com.iprd.rdtcamera.Utils.rotateFrame(greyMat, -90);
                org.opencv.core.Size sz = new org.opencv.core.Size(1280, 720);
                Imgproc.resize(mSetRotation ? rotatedmat:greyMat, greyMatResized, sz, 0.0, 0.0, INTER_CUBIC);
                mPreProcessingTime = System.currentTimeMillis() - st;

                if( mRDTProcessingResultAvailable) {
                    if ((mStatus!= null) && mStatus.mRDTFound) {
                        //Find Transformation..
                        ret.mRDTFound =true;
                        ret.mBoundingBoxX = mStatus.mBoundingBoxX;
                        ret.mBoundingBoxY = mStatus.mBoundingBoxY;
                        ret.mBoundingBoxWidth = mStatus.mBoundingBoxWidth;
                        ret.mBoundingBoxHeight = mStatus.mBoundingBoxHeight;
                        warpmat = ImageRegistration.FindMotionRefIns(greyMat,mGreyMat);
                        if(warpmat!=null) {
                            //Log.i("Tx-Ty", warpmat.get(0, 2)[0] + "x" + warpmat.get(1, 2)[0]);
                            if (!((Math.abs(warpmat.get(0, 2)[0]) > mConfig.mMaxAllowedTranslationX || Math.abs(warpmat.get(1, 2)[0]) > mConfig.mMaxAllowedTranslationY))) {
                                lt = warpPoint(new Point(mStatus.mBoundingBoxX / 16.0, mStatus.mBoundingBoxY / 16.0), warpmat);
                                lt.x = lt.x * 16;
                                lt.y = lt.y * 16;
                                ClampBoundingBox(mGreyMat, lt);
                                ret.mBoundingBoxX = (short) lt.x;
                                ret.mBoundingBoxY = (short) lt.y;
                                ret.mBoundingBoxWidth = (short)((ret.mBoundingBoxWidth + ret.mBoundingBoxX)> greyMat.cols()? greyMat.cols() - ret.mBoundingBoxX :ret.mBoundingBoxWidth);////(short)((rb.x-lt.x)*16);
                                ret.mBoundingBoxHeight = (short)((ret.mBoundingBoxHeight + ret.mBoundingBoxY)> greyMat.rows()? greyMat.rows() - ret.mBoundingBoxY:ret.mBoundingBoxHeight);
                            }
                        }
                    }else{
                        ret.mRDTFound =false;
                    }
                    if(mInputMat!= null)mInputMat.release();
                    if(mGreyMat!= null)mGreyMat.release();
                    if(mGreyMatResized!= null)mGreyMatResized.release();
                    mRDTProcessingResultAvailable=false;
                }
                //We should thread from here
                if (!mRDTProcessing) {
                    mRDTProcessing = true;
                    mRDTProcessingResultAvailable = false;
                    mInputMat = matinput.clone();
                    mGreyMat = greyMat.clone();
                    mGreyMatResized = greyMatResized.clone();
                    mStatus = new AcceptanceStatus();
                    //Lets run it as thread
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            ProcessRDT(mStatus, mInputMat, mGreyMatResized);
                        }
                    }).start();
                }
            }

            if(mPlaybackMode) {
                if(ret.mRDTFound)rectangle(matinput, new Point(ret.mBoundingBoxX, ret.mBoundingBoxY), new Point(ret.mBoundingBoxX +ret.mBoundingBoxWidth, ret.mBoundingBoxY+ret.mBoundingBoxHeight), new Scalar(255, 0, 0, 0), 4, LINE_AA, 0);
                mLocalcopy = matinput.clone();
            }
            if (!ret.mRDTFound) return ret;

            //Log.d("Bounding Box Used ",ret.mBoundingBoxX+"x"+ret.mBoundingBoxY +"  "+ret.mBoundingBoxWidth+"x"+ret.mBoundingBoxHeight);// greyMatResized.submat(detectedRoi);)
            ismPreviousRDT = true;
            mPreviousLTPoint = new Point(ret.mBoundingBoxX,ret.mBoundingBoxY);
            mPreviousRBPoint = new Point(ret.mBoundingBoxWidth,ret.mBoundingBoxHeight);
            Mat imageROI = greyMat.submat(new Rect(ret.mBoundingBoxX,ret.mBoundingBoxY,ret.mBoundingBoxWidth,ret.mBoundingBoxHeight));// greyMatResized.submat(detectedRoi);
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
            greyMatResized.release();
            greyMat.release();
            matinput.release();
            mInprogress = false;
            mPostProcessingTime = System.currentTimeMillis()-mPostProcessingTime;
        }
        return ret;
    }

    private void ClampBoundingBox(Mat greyMat, Point lt) {
        if(lt.x < 0.0) lt.x=0;
        if(lt.x > greyMat.cols()) lt.x=greyMat.cols();
        if(lt.y < 0.0) lt.y=0;
        if(lt.y > greyMat.rows()) lt.y=greyMat.rows();
    }

    private void ProcessRDT(AcceptanceStatus retStatus,Mat inputmat,Mat reszgreymat){
        Rect detectedRoi=null;
        try {
            //Log.i("ProcessRDT","Coming in process RDT");
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
                //Log.d("Bounding Box Computed ",retStatus.mBoundingBoxX+"x"+retStatus.mBoundingBoxY +"  "+retStatus.mBoundingBoxWidth+"x"+retStatus.mBoundingBoxHeight);// greyMatResized.submat(detectedRoi);)
            }
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
        finally {
            mRDTProcessingResultAvailable=true;
            mRDTProcessing = false;
            //Log.i("ProcessRDT","Exiting in process RDT");
        }
        //return detectedRoi;
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