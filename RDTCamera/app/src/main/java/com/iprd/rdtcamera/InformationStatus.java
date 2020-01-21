package com.iprd.rdtcamera;

import android.graphics.Bitmap;

public class InformationStatus{
    public Bitmap mTrackedImage;
    public Bitmap mWarpedImage;
    public InformationStatus(){
        mTrackedImage=null;
        mWarpedImage = null;
        mBrightness = mSharpness = 0;
        mMinError = -1;
        mAngle=-1;
        mScale=-1;
    }
    public int mBrightness;
    public int mSharpness;
    public double mMinError;
    public double mAngle;
    public double mScale;
}
