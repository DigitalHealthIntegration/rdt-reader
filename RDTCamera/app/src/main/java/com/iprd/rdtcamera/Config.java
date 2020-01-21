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
    public short mMaxFrameTranslationalMagnitude;
    public short mMax10FrameTranslationalMagnitude;
    public double mMinRdtErrorThreshold;
    public short mMinSteadyFrames;
    public Config() {
        setDefaults();
    }


    public void setmMinSteadyFrames(short mMinSteadyFrames) {
        this.mMinSteadyFrames = mMinSteadyFrames;
    }

    public void setDefaults() {
        mMaxScale = 95; //% of width for bb width
        mMinScale = 5;//% of width for bb width
        mXMin = 15; //% of width
        mXMax = 85; //% of width
        mYMin = 0; //% of HEIGHT
        mYMax = 100; //% of HEIGHT
        mMinSharpness = 800.0f;
        mMaxBrightness = 220.0f;
        mMinBrightness = 110.0f;
        mTfliteB=null;
        mMappedByteBuffer=null;
        mMaxAllowedTranslationY = 6; //level 4
        mMaxAllowedTranslationX = 6; //level 4
        mMaxFrameTranslationalMagnitude = 30;
        mMax10FrameTranslationalMagnitude =200;
    }
    public void setmMappedByteBuffer(MappedByteBuffer mMappedByteBuffer) {
        this.mMappedByteBuffer = mMappedByteBuffer;
    }

    public void setmMinRdtErrorThreshold(double mMinRdtErrorThreshold) {
        this.mMinRdtErrorThreshold = mMinRdtErrorThreshold;
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

    public void setMaxFrameTranslationalMagnitude(short mMaxFrameTranslationalMagnitude) {
        this.mMaxFrameTranslationalMagnitude = mMaxFrameTranslationalMagnitude;
    }

    public void setMax10FrameTranslationalMagnitude(short mMax10FrameTranslationalMagnitude) {
        this.mMax10FrameTranslationalMagnitude = mMax10FrameTranslationalMagnitude;
    }

}

