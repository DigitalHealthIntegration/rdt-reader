package com.iprd.rdtcamera;

import java.nio.MappedByteBuffer;

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
    public MappedByteBuffer mMappedByteBuffer;
    public int mMaxAllowedTranslationX;
    public int mMaxAllowedTranslationY;

    public Config() {
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
        mTfliteB=null;
        mMappedByteBuffer=null;
        mMaxAllowedTranslationY = 400;
        mMaxAllowedTranslationX = 200;
    }
    public void setmMappedByteBuffer(MappedByteBuffer mMappedByteBuffer) {
        this.mMappedByteBuffer = mMappedByteBuffer;
    }

    public void setmTfliteB(byte[] mTfliteB) {
        this.mTfliteB = mTfliteB;
    }

    public void setmMinBrightness(float mMinBrightness) {
        this.mMinBrightness = mMinBrightness;
    }

    public void setmMaxBrightness(float mMaxBrightness) {
        this.mMaxBrightness = mMaxBrightness;
    }

    public void setmMinSharpness(float mMinSharpness) {
        this.mMinSharpness = mMinSharpness;
    }

    public void setmYMax(short mYMax) {
        this.mYMax = mYMax;
    }

    public void setmYMin(short mYMin) {
        this.mYMin = mYMin;
    }

    public void setmXMax(short mXMax) {
        this.mXMax = mXMax;
    }

    public void setmXMin(short mXMin) {
        this.mXMin = mXMin;
    }

    public void setmMinScale(short mMinScale) {
        this.mMinScale = mMinScale;
    }

    public void setmMaxScale(short mMaxScale) {
        this.mMaxScale = mMaxScale;
    }

    public void setmMaxAllowedTranslationX(int mMaxAllowedTranslationX) {
        this.mMaxAllowedTranslationX = mMaxAllowedTranslationX;
    }
    public void setmMaxAllowedTranslationY(int mMaxAllowedTranslationY) {
        this.mMaxAllowedTranslationY = mMaxAllowedTranslationY;
    }

}

