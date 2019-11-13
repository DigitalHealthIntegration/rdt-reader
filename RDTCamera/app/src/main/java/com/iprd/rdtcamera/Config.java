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
    public short mMaxAllowedTranslationX;
    public short mMaxAllowedTranslationY;

    public Config() {
        setDefaults();
    }

    public void setDefaults() {
        mMaxScale = 75;
        mMinScale = 5;
        mXMin = 15;
        mXMax = 75;
        mYMin = 10;
        mYMax = 75;
        mMinSharpness = 800.0f;
        mMaxBrightness = 220.0f;
        mMinBrightness = 110.0f;
        mTfliteB=null;
        mMappedByteBuffer=null;
        mMaxAllowedTranslationY = 6; //level 4
        mMaxAllowedTranslationX = 6; //level 4
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

    public void setmMaxAllowedTranslationX(short mMaxAllowedTranslationX) {
        this.mMaxAllowedTranslationX = mMaxAllowedTranslationX;
    }
    public void setmMaxAllowedTranslationY(short mMaxAllowedTranslationY) {
        this.mMaxAllowedTranslationY = mMaxAllowedTranslationY;
    }

}

