package com.iprd.rdtcamera;


import java.nio.MappedByteBuffer;

class Config{
    Config() {
        mMaxScale = 1100;
        mMinScale = 700;
        mXMin = 100;
        mXMax = 500;
        mYMin = 50;
        mYMax = 650;
        mMinSharpness = 500.0f;
        mMaxBrightness = 210.0f;
        mMinBrightness = 110.0f;
    }
    short mMaxScale;
    short mMinScale;
    short mXMin;
    short mXMax;
    short mYMin;
    short mYMax;
    float mMinSharpness;
    float mMaxBrightness;
    float mMinBrightness;
    byte[] mTfliteB;
}

class AcceptanceStatus {
    public static final short NOT_COMPUTED = 100;
    public static final short TOO_HIGH = 1;
    public static final short TOO_LOW = -1;
    public static final short GOOD = 0;

    AcceptanceStatus() {
        setDefaultStatus();
    }
    AcceptanceStatus(short sharp,short scale,short bright,short prosp,short dispx,short dispy,short x,short y, short width, short height){
        mRDTFound = true;
        mBrightness = bright;
        mSharpness = sharp;
        mScale = scale;
        mDisplacementX = dispx;
        mDisplacementY = dispy;
        mPerspectiveDistortion = prosp;
        mBoundingBoxX = x;
        mBoundingBoxY = y;
        mBoundingBoxWidth = width;
        mBoundingBoxHeight = height;
    }

    short mSharpness;
    short mScale;
    short mBrightness;
    short mPerspectiveDistortion;
    short mDisplacementX; //displacement of the RDT center.x from ideal
    short mDisplacementY; //displacement of the RDT center.y from ideal
    boolean mRDTFound;// was an RDT found by the coarse finder
    short mBoundingBoxX, mBoundingBoxY;
    short mBoundingBoxWidth, mBoundingBoxHeight;

    void setDefaultStatus(){
        mRDTFound = false;
        mBrightness = NOT_COMPUTED;
        mSharpness = NOT_COMPUTED;
        mScale = NOT_COMPUTED;
        mDisplacementX = NOT_COMPUTED;
        mDisplacementY = NOT_COMPUTED;
        mPerspectiveDistortion = NOT_COMPUTED;
        mBoundingBoxX = mBoundingBoxY= mBoundingBoxWidth=mBoundingBoxHeight=-1;
    }
};