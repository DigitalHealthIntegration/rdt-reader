package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.MappedByteBuffer;
import java.util.Vector;

import static com.iprd.rdtcamera.AcceptanceStatus.GOOD;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_HIGH;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_LOW;
import static com.iprd.rdtcamera.CvUtils.PrintAffineMat;
import static com.iprd.rdtcamera.CvUtils.mComputeVector_FinalMVector;
import static com.iprd.rdtcamera.CvUtils.mComputeVector_FinalPoint;
import static com.iprd.rdtcamera.CvUtils.scaleAffineMat;
import static com.iprd.rdtcamera.CvUtils.warpPoint;
import static com.iprd.rdtcamera.ImageRegistration.ComputeMotion;
import static com.iprd.rdtcamera.ImageRegistration.FindMotionRefIns;
import static com.iprd.rdtcamera.Utils.SaveMatrix;
import static com.iprd.rdtcamera.Utils.getBitmapFromMat;
import static com.iprd.rdtcamera.Utils.rotateRect;
import static org.opencv.core.Core.BORDER_REFLECT101;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.core.Core.LINE_AA;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.meanStdDev;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.Laplacian;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.line;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.pyrDown;
import static org.opencv.imgproc.Imgproc.rectangle;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.imgproc.Imgproc.warpPerspective;

public class RdtAPI {
    private Config mConfig;
    private AcceptanceStatus mAcceptanceStatus;//=new AcceptanceStatus();
    private ObjectDetection mTensorFlow;//=null;
    private static volatile boolean mInprogress = false;
    boolean mLinearflow=false;
    private static volatile boolean mRDTProcessing = false;
    private static volatile boolean mRDTProcessingResultAvailable = false;
    Mat mInputMat;
    Mat mGreyMat;
    Mat mGreyMatResized;
    AcceptanceStatus mStatus=null;
    Vector<Pair<Mat,Mat>> mWarpList= new Vector<>();
    Mat mRefImage=null;
    int mRefCount=0;
    long mPreviousTime=0;
    private short mBrightness;
    private short mSharpness;
    Mat mLocalcopy;
    boolean mPlaybackMode;
    long mTensorFlowProcessTime;
    long mPreProcessingTime;
    long mPostProcessingTime;
    Mat mPipMat=null;
    Mat mMotionVectorMat=null;
    Mat mWarpedMat=null;
    Mat mWarpInfo=null;
    private final Object piplock = new Object();

    public void setmShowPip(boolean mShowPip) {
        this.mShowPip = mShowPip;
    }
    public Bitmap getPipMat(){
        synchronized (piplock) {
            if (mPipMat == null) return null;
            return getBitmapFromMat(mPipMat);
        }
    }

    boolean mShowPip = false;

    public void setLinearflow(boolean linearflow){
        this.mLinearflow = linearflow;
    }


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
        //setRotation(false);
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
        ret.mInfo.mSharpness = (int)sharpness;

        //Log.d("Sharpness","mSharpness " + sharpness);
        if (sharpness < mConfig.mMinSharpness){
            ret.mSharpness = TOO_LOW;
            return false;
        }
        Log.d("Sharpness",""+sharpness);
        ret.mSharpness = AcceptanceStatus.GOOD;
        return true;
    }

    private boolean computeBrightness(Mat grey,AcceptanceStatus ret) {
        Scalar tempVal = mean(grey);
        double brightness = tempVal.val[0];
        mBrightness = (short) brightness;
        ret.mInfo.mBrightness = mBrightness;
        //Log.d("Brightness","mBrightness "+brightness);
        if (brightness > mConfig.mMaxBrightness) {
            ret.mBrightness = TOO_HIGH;
            return false;
        }else if (brightness < mConfig.mMinBrightness){
            ret.mBrightness = TOO_LOW;
            return false;
        }
        Log.d("Brightness",""+mBrightness);
        ret.mBrightness = AcceptanceStatus.GOOD;
        return true;
    }

    private boolean computeDistortion(Mat mat,AcceptanceStatus ret){
        if(mSetRotation) {
            if (ret.mBoundingBoxWidth * 100 > mConfig.mMaxScale * mat.cols()) {
                ret.mScale = TOO_HIGH;
                Log.d("DistMaxSc", "" + ret.mBoundingBoxWidth * 100 + ">" + mConfig.mMaxScale * mat.cols());
                return false;
            } else if (ret.mBoundingBoxWidth * 100 < mConfig.mMinScale * mat.cols()) {
                ret.mScale = TOO_LOW;
                Log.d("DistMinSc", "" + ret.mBoundingBoxWidth * 100 + "<" + mConfig.mMinScale * mat.cols());
                return false;
            } else ret.mScale = AcceptanceStatus.GOOD;

            if (ret.mBoundingBoxX*100 > mConfig.mXMax*mat.cols()){
                ret.mDisplacementX = TOO_HIGH;
                Log.d("DistXMax",""+ret.mBoundingBoxX*100 +">"+ mConfig.mXMax*mat.cols());
                return false;
            }else if (ret.mBoundingBoxX*100 < mConfig.mXMin*mat.cols()){
                ret.mDisplacementX = TOO_LOW;
                Log.d("DistXMin",""+ret.mBoundingBoxX *100+"<"+ mConfig.mXMin*mat.cols());
                return false;
            }else ret.mDisplacementX = GOOD;

            if (ret.mBoundingBoxY*100 > mConfig.mYMax*mat.rows()){
                ret.mDisplacementY = TOO_HIGH;
                Log.d("DistYMax",""+ret.mBoundingBoxY*100 +">"+ mConfig.mYMax*mat.rows());
                return false;
            }else if (ret.mBoundingBoxY*100 < mConfig.mYMin*mat.rows()){
                ret.mDisplacementY = TOO_LOW;
                Log.d("DistYMin",""+ret.mBoundingBoxY*100 +"<"+ mConfig.mYMin*mat.rows());
                return false;
            }else ret.mDisplacementY = GOOD;

        }else{
            if (ret.mBoundingBoxHeight * 100 > mConfig.mMaxScale * mat.rows()) {
                ret.mScale = TOO_HIGH;
                Log.d("DistMaxSc", "" + ret.mBoundingBoxHeight * 100 + ">" + mConfig.mMaxScale * mat.rows());
                return false;
            } else if (ret.mBoundingBoxHeight * 100 < mConfig.mMinScale * mat.rows()) {
                ret.mScale = TOO_LOW;
                Log.d("DistMinSc", "" + ret.mBoundingBoxHeight * 100 + "<" + mConfig.mMinScale * mat.rows());
                return false;
            } else ret.mScale = AcceptanceStatus.GOOD;

            if (ret.mBoundingBoxX*100 > mConfig.mXMax*mat.rows()){
                ret.mDisplacementX = TOO_HIGH;
                Log.d("DistXMax",""+ret.mBoundingBoxX*100 +">"+ mConfig.mXMax*mat.rows());
                return false;
            }else if (ret.mBoundingBoxX*100 < mConfig.mXMin*mat.rows()){
                ret.mDisplacementX = TOO_LOW;
                Log.d("DistXMin",""+ret.mBoundingBoxX *100+"<"+ mConfig.mXMin*mat.rows());
                return false;
            }else ret.mDisplacementX = GOOD;

            if (ret.mBoundingBoxY*100 > mConfig.mYMax*mat.cols()){
                ret.mDisplacementY = TOO_HIGH;
                Log.d("DistYMax",""+ret.mBoundingBoxY*100 +">"+ mConfig.mYMax*mat.cols());
                return false;
            }else if (ret.mBoundingBoxY*100 < mConfig.mYMin*mat.cols()){
                ret.mDisplacementY = TOO_LOW;
                Log.d("DistYMin",""+ret.mBoundingBoxY*100 +"<"+ mConfig.mYMin*mat.cols());
                return false;
            }else ret.mDisplacementY = GOOD;
        }

        ret.mPerspectiveDistortion= GOOD;
        return true;
    }

    public AcceptanceStatus checkFrame(Bitmap capFrame) {
        mInprogress = true;
        mBrightness = -1;
        mSharpness = -1;
        Mat matinput = new Mat();
        Mat greyMat = new Mat();
        Mat greyMatResized = new Mat();
        mMotionVectorMat = null;
        Mat mPrevious;
        mPreProcessingTime = mTensorFlowProcessTime = mPostProcessingTime= 0;

        AcceptanceStatus ret= new AcceptanceStatus();
        try {
            long st  = System.currentTimeMillis();
            Utils.bitmapToMat(capFrame, matinput);
            if (false) SaveMatrix(matinput, "Input");

            cvtColor(matinput, greyMat, Imgproc.COLOR_RGBA2GRAY);
            Point lt = null,rb=null;
            Mat warp = null;

            warp = ComputeMotion(greyMat);
            Pair<Mat,Mat> item = new Pair<>(warp.clone(),greyMat.clone());
            mWarpList.add(item);

            long stend  = System.currentTimeMillis()-st;
            Log.d("MotionComp",stend+"");
            if(mWarpList.size() >10){
                mWarpList.elementAt(0).first.release();
                mWarpList.elementAt(0).second.release();
                mWarpList.remove(0);
            }
            //Lets Check Motion now.
            if(mRefImage==null) {
                mRefImage = greyMat.clone();
                mWarpInfo = Mat.eye(2,3,CV_32F);
            }
            Point p = new Point(0,0);
            for(int i=1;i<mWarpList.size();i++){
                p.x +=mWarpList.elementAt(i).first.get(0,2)[0];
                p.y +=mWarpList.elementAt(i).first.get(1,2)[0];
                //Log.i(i+"", mWarpList.elementAt(i).get(0,2)[0] + "x" + mWarpList.elementAt(i).get(1,2)[0]);
            }
            Mat warp10 = FindMotionRefIns(mRefImage,greyMat,mWarpInfo,false);
            mWarpedMat = new Mat(greyMat.width(), greyMat.height(), greyMat.type());
            warpAffine(greyMat,mWarpedMat,warp10,greyMat.size());
            Imgproc.resize(mWarpedMat, mWarpedMat, new Size(mWarpedMat.width()>>2,mWarpedMat.height()>>2), 0.0, 0.0, INTER_CUBIC);
//

            long currtime  = System.currentTimeMillis();
            if((currtime - mPreviousTime > 2*1000)||(Math.abs(warp10.get(0,2)[0]) >= (greyMat.width()>>2))||(Math.abs(warp10.get(1,2)[0]) >= (greyMat.height()>>2))){
                Log.i("RDT Ref","Reset Ref ");
                mRefCount = 0;
                mPreviousTime=currtime;
                mRefImage.release();
                mWarpInfo.release();
                mRefImage = greyMat.clone();
                mWarpInfo = Mat.eye(2,3,CV_32F);
                putText(mWarpedMat, "RDT REFERENCE IMAGE ", new Point(0, mWarpedMat.cols()>>1), FONT_HERSHEY_SIMPLEX, 2.0,new Scalar(255,0,0,0),5);
            }

            //Threshold1 and Threshold 2.
            Log.i("10Comp frame", warp10.get(0,2)[0] + "x" + warp10.get(1,2)[0]);
            Log.i("10Add frame", p.x + "x" + p.y);
            Log.i("1 frame", warp.get(0,2)[0] + "x" + warp.get(1,2)[0]);

            Scalar srg = new Scalar(255,255,0,0);//RGBA
            mMotionVectorMat = CvUtils.ComputeVector(new Point(warp10.get(0,2)[0],warp10.get(1,2)[0]),mMotionVectorMat,srg);

            Scalar sr = new Scalar(255,0,0,0);//RGBA
            mMotionVectorMat = CvUtils.ComputeVector(p,mMotionVectorMat,sr);
            ret.mSteady = GOOD;
            if(mComputeVector_FinalMVector.x > mConfig.mMax10FrameTranslationalMagnitude){
                ret.mSteady = TOO_HIGH;
            }

            Scalar sg = new Scalar(0,255,0,0);
            mMotionVectorMat = CvUtils.ComputeVector(new Point(warp.get(0,2)[0],warp.get(1,2)[0]),mMotionVectorMat,sg);
            if(mComputeVector_FinalMVector.x > mConfig.mMaxFrameTranslationalMagnitude){
                ret.mSteady = TOO_HIGH;
            }
            //process frame
            if(matinput.width() < matinput.height()) {
                mSetRotation = true;
            }
            Rect detectedRoi = null;
            Mat rotatedmat = new Mat();
            if (mSetRotation) rotatedmat = com.iprd.rdtcamera.Utils.rotateFrame(greyMat, -90);
            org.opencv.core.Size sz = new org.opencv.core.Size(1280, 720);
            Imgproc.resize(mSetRotation ? rotatedmat:greyMat, greyMatResized, sz, 0.0, 0.0, INTER_CUBIC);
            mPreProcessingTime = System.currentTimeMillis() - st;
            //mRDTProcessingResultAvailable=false;
            if( mRDTProcessingResultAvailable) {
                if ((mStatus!= null) && mStatus.mRDTFound) {
                    //Find Transformation..
                    ret.mRDTFound =true;
                    Log.i("Rect ",mStatus.mBoundingBoxX+"x"+mStatus.mBoundingBoxY+" "+mStatus.mBoundingBoxWidth+"x"+mStatus.mBoundingBoxHeight);
                    ret.mBoundingBoxX = mStatus.mBoundingBoxX;
                    ret.mBoundingBoxY = mStatus.mBoundingBoxY;
                    ret.mBoundingBoxWidth = mStatus.mBoundingBoxWidth;
                    ret.mBoundingBoxHeight = mStatus.mBoundingBoxHeight;
                }else{
                    ret.mRDTFound =false;
                }
                if(mInputMat != null ) mInputMat.release();
                if(mGreyMat != null) mGreyMat.release();
                if(mGreyMatResized != null) mGreyMatResized.release();
                mRDTProcessingResultAvailable=false;
            }
            //We should thread from here
            if (!mRDTProcessing) {
                mRDTProcessing = true;
                mRDTProcessingResultAvailable = false;
                if (mInputMat != null) mInputMat.release();
                if (mGreyMat != null) mGreyMat.release();
                if (mGreyMatResized != null) mGreyMatResized.release();
                mInputMat = matinput.clone();
                mGreyMat = greyMat.clone();
                mGreyMatResized = greyMatResized.clone();
                mStatus = new AcceptanceStatus();
                mStatus.mSteady = ret.mSteady;
                if (mLinearflow) {
                    ProcessRDT(mStatus, mInputMat, mGreyMatResized);
                    ret = mStatus;
                } else {
                    //Lets run it as thread
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            ProcessRDT(mStatus, mInputMat, mGreyMatResized);
                        }
                    }).start();
                }
            }
            mPostProcessingTime = System.currentTimeMillis();
            if(mPlaybackMode) {
                if(ret.mRDTFound)rectangle(matinput, new Point(ret.mBoundingBoxX, ret.mBoundingBoxY), new Point(ret.mBoundingBoxX +ret.mBoundingBoxWidth, ret.mBoundingBoxY+ret.mBoundingBoxHeight), new Scalar(255, 0, 0, 0), 4, LINE_AA, 0);
                mLocalcopy = matinput.clone();
            }
            if (!ret.mRDTFound) return ret;

            //Log.d("Bounding Box Used ",ret.mBoundingBoxX+"x"+ret.mBoundingBoxY +"  "+ret.mBoundingBoxWidth+"x"+ret.mBoundingBoxHeight);// greyMatResized.submat(detectedRoi);)
            if(!computeDistortion(greyMat,ret)){
                return ret;
            }
            Mat imageROI = greyMat.submat(new Rect(ret.mBoundingBoxX,ret.mBoundingBoxY,ret.mBoundingBoxWidth,ret.mBoundingBoxHeight));// greyMatResized.submat(detectedRoi);
            //SaveMatrix(imageROI,"ROI");
            if (!computeBlur(imageROI,ret)) {
                return ret;
            }
            if (!computeBrightness(imageROI,ret)) {
                return ret;
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            Log.i("BB ",ret.mBoundingBoxX+"x"+ret.mBoundingBoxY+"-"+ret.mBoundingBoxWidth+"x"+ret.mBoundingBoxHeight);
            greyMatResized.release();
            greyMat.release();
            matinput.release();
            ret.mInfo.mWarpedImage = getBitmapFromMat(mWarpedMat);
            ret.mInfo.mMinError = mTensorFlow.getMinError();
            if(mWarpedMat != null) mWarpedMat.release();
            if(mMotionVectorMat!= null){
                ret.mInfo.mTrackedImage = getBitmapFromMat(mMotionVectorMat);
                mMotionVectorMat.release();
                mMotionVectorMat=null;
            }
            mInprogress = false;
            mPostProcessingTime = System.currentTimeMillis()-mPostProcessingTime;
        }
        return ret;
    }



    private void UpdateBB(Mat greyMat, AcceptanceStatus ret, Point lt, double x, int cols, double y, int rows) {
        ClampBoundingBox(greyMat, lt);
        ret.mBoundingBoxX = (short) (lt.x);
        ret.mBoundingBoxY = (short) (lt.y);
        ret.mBoundingBoxWidth = (short) ((x + ret.mBoundingBoxX) >= cols ? cols - ret.mBoundingBoxX : x);////(short)((rb.x-lt.x)*16);
        ret.mBoundingBoxHeight = (short) ((y + ret.mBoundingBoxY) >= rows ? rows - ret.mBoundingBoxY : y);
    }

    private void ClampBoundingBox(Mat greyMat, Point lt) {
        if(lt.x < 0.0) lt.x=0;
        if(lt.x >= greyMat.cols()) lt.x=greyMat.cols();
        if(lt.y < 0.0) lt.y=0;
        if(lt.y >= greyMat.rows()) lt.y=greyMat.rows();
    }

    private void ProcessRDT(AcceptanceStatus retStatus,Mat inputmat,Mat reszgreymat){
        Rect detectedRoi=null;
        final long updatetimest = System.currentTimeMillis();

        try {
            //Log.i("ProcessRDT","Coming in process RDT");
            Boolean[] rdtFound = new Boolean[]{new Boolean(false)};
            detectedRoi = mTensorFlow.update(reszgreymat, rdtFound);
            mTensorFlowProcessTime = System.currentTimeMillis() - updatetimest;

            retStatus.mRDTFound = rdtFound[0].booleanValue();
            if (retStatus.mRDTFound) {
                Rect roi = detectedRoi.clone();
                if (mSetRotation) {
                    roi = rotateRect(reszgreymat, roi, -90);
                }
                float wfactor = 0;
                float hfactor = 0;
                if (!mSetRotation) {
                    wfactor = inputmat.cols() / 1280.f;
                    hfactor = inputmat.rows() / 720f;
                } else {
                    wfactor = inputmat.cols() / 720.0f;
                    hfactor = inputmat.rows() / 1280f;
                }
                //handle rotation TBD
                if (mSaveInput || mPlaybackMode || mShowPip) {
                    rectangle(inputmat, new Point(roi.x * wfactor, roi.y * hfactor), new Point((roi.x + roi.width) * wfactor, (roi.y + roi.height) * hfactor), new Scalar(255, 128, 255, 0), 8, LINE_AA, 0);
                }
                if(mShowPip){
                    synchronized (piplock) {
                        int w = inputmat.width()>>2,h = inputmat.height()>>2;
                        if(!mSetRotation){
                            w=inputmat.height()>>2;
                            h=inputmat.width()>>2;
                        }
                        mPipMat = new Mat(w, h, inputmat.type());
                        resize(inputmat, mPipMat, new Size(w, h));
                    }
                }
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
            Log.i("Process RDT Time ", ""+(System.currentTimeMillis()-updatetimest));
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

        public RdtAPIBuilder setmMaxAllowedTranslationX(short mX) {
            mConfig.setmMaxAllowedTranslationX(mX);
            return this;
        }

        public RdtAPIBuilder setmMaxAllowedTranslationY(short mY) {
            mConfig.setmMaxAllowedTranslationY(mY);
            return this;
        }

        public RdtAPIBuilder setMaxFrameTranslationalMagnitude(short mMaxFrameTranslationalMagnitude) {
            mConfig.setMaxFrameTranslationalMagnitude(mMaxFrameTranslationalMagnitude);
            return this;
        }

        public RdtAPIBuilder setMax10FrameTranslationalMagnitude(short mMax10FrameTranslationalMagnitude) {
            mConfig.setMax10FrameTranslationalMagnitude(mMax10FrameTranslationalMagnitude);
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