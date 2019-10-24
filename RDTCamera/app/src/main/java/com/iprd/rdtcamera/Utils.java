package com.iprd.rdtcamera;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.util.Log;

import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class Utils {

    static String dirpath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/RDT/";
    static int mImageCount = 0;

    static void  CreateDirectory() {

        File file = new File(dirpath);
        if (!file.exists()) {
            file.mkdirs();
        }
    }

    public static void SaveROIImage(Mat greyMat, int x1, int y1, int x2, int y2) {
        Mat tmp = new Mat();
        Imgproc.cvtColor(greyMat, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
        Imgproc.rectangle(tmp, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255), 1);
        Bitmap finalBitmap = null;
        try {
            //Imgproc.cvtColor(seedsImage, tmp, Imgproc.COLOR_RGB2BGRA);
            finalBitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(tmp, finalBitmap);
            saveImage(finalBitmap,"Grey");
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        tmp.release();
    }

    public static void saveImage(Bitmap m,String suff) {
        CreateDirectory();
        File myImage = new File(dirpath+"Image" + mImageCount+suff + ".jpg");
        Log.i("Saving File",myImage.getAbsolutePath());
        mImageCount++;
        if (myImage.exists()) myImage.delete();
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(myImage);
            m.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static Short ApplySettings(Context c,RdtAPI rdtapi) {
        double mTopTh = ObjectDetection.mTopThreshold;
        double mBotTh = ObjectDetection.mBottomThreshold;
        Short mShowImageData=0;
        Config config= new Config();
        boolean mSaveNegativeData= false;
        try {
            SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(c);
            config.mMaxScale = Short.parseShort(prefs.getString("mMaxScale",  config.mMaxScale+""));
            config.mMinScale = Short.parseShort(prefs.getString("mMinScale", config.mMinScale+""));
            config.mXMin = Short.parseShort(prefs.getString("mXMin",  config.mXMin+""));
            config.mXMax = Short.parseShort(prefs.getString("mXMax", config.mXMax+""));
            config.mYMin = Short.parseShort(prefs.getString("mYMin", config.mYMin+""));
            config.mYMax = Short.parseShort(prefs.getString("mYMax", config.mYMax+""));
            config.mMinSharpness = Float.parseFloat(prefs.getString("mMinSharpness", config.mMinSharpness +""));
            config.mMaxBrightness = Float.parseFloat(prefs.getString("mMaxBrightness", config.mMaxBrightness+""));
            config.mMinBrightness = Float.parseFloat(prefs.getString("mMinBrightness", config.mMinBrightness+""));
            mTopTh = Float.parseFloat(prefs.getString("mTopTh", ObjectDetection.mTopThreshold+""));
            mBotTh = Float.parseFloat(prefs.getString("mBotTh", ObjectDetection.mBottomThreshold+""));
            mShowImageData  = Short.parseShort(prefs.getString("mShowImageData", "0"));
            short t  = Short.parseShort(prefs.getString("mSaveNegativeData", mSaveNegativeData?"1":"0"));
            if(t!=0) mSaveNegativeData =true;
        }catch (NumberFormatException nfEx){//prefs.getString("mMinBrightness", "110.0f")
            Log.i("RDT","Exception in  Shared Pref switching to default");
            config.setDefaults();
            mTopTh = ObjectDetection.mTopThreshold;
            mBotTh = ObjectDetection.mBottomThreshold;
            mShowImageData = 0;
            mSaveNegativeData = false;
        }
        rdtapi.setConfig(config);
        rdtapi.setTopThreshold(mTopTh);
        rdtapi.setBottomThreshold(mBotTh);
        rdtapi.mSaveNegativeData = mSaveNegativeData;
        return mShowImageData;
    }
}
