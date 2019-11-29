package com.iprd.rdtcamera;

import android.graphics.Bitmap;

public class InformationStatus{
    public Bitmap mTrackedImage;
    public InformationStatus(){
        mTrackedImage=null;
        mBrightness = mSharpness = 0;
        mMinRdtError= 100.0;
    }
    public int mBrightness;
    public int mSharpness;
    public double mMinRdtError;

}
