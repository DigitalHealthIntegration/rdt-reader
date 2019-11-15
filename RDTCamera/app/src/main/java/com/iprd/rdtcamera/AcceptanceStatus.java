package com.iprd.rdtcamera;

import android.graphics.Bitmap;

import org.opencv.core.Mat;


public class AcceptanceStatus {
    public static final short NOT_COMPUTED = 100;
    public static final short TOO_HIGH = 1;
    public static final short TOO_LOW = -1;
    public static final short GOOD = 0;

    public short mSharpness;
    public short mScale;
    public short mBrightness;
    public short mPerspectiveDistortion;
    public short mDisplacementX; //displacement of the RDT center.x from ideal
    public short mDisplacementY; //displacement of the RDT center.y from ideal
    public boolean mRDTFound;// was an RDT found by the coarse finder
    public short mBoundingBoxX, mBoundingBoxY;
    public short mBoundingBoxWidth, mBoundingBoxHeight;
    public short mSteady;

    public InformationStatus mInfo;
    public String GetResult(){
        String s = mRDTFound?"1":"0"+","+mBoundingBoxX+"x"+mBoundingBoxY+"-"+mBoundingBoxWidth+"x"+mBoundingBoxHeight+","+mBrightness +","+mSharpness+","+mScale
                +","+mDisplacementX +","+mDisplacementY+","+mPerspectiveDistortion+","+mSteady;
        return s;
    }
    AcceptanceStatus() {
        setDefaultStatus();
    }

    void setDefaultStatus(){
        mRDTFound = false;
        mBrightness = NOT_COMPUTED;
        mSharpness = NOT_COMPUTED;
        mScale = NOT_COMPUTED;
        mDisplacementX = NOT_COMPUTED;
        mDisplacementY = NOT_COMPUTED;
        mPerspectiveDistortion = NOT_COMPUTED;
        mBoundingBoxX = mBoundingBoxY= mBoundingBoxWidth=mBoundingBoxHeight=-1;
        mSteady = NOT_COMPUTED;
        mInfo=new InformationStatus();
    }
};