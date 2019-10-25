package com.iprd.rdtcamera;

public class Config{
    public short mMaxScale;
    public short mMinScale;
    public short mXMin;
    public short mXMax;
    public short mYMin;
    public short mYMax;
    public float mMinSharpness;
    public float mMaxBrightness;
    public float mMinBrightness;
    public byte[] mTfliteB;

    public Config
            () {
        setDefaults();
    }

    public void setDefaults() {
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
}

