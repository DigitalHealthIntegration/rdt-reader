package com.iprd.rdtcamera;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.preference.Preference;
import android.preference.PreferenceManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.prefs.Preferences;

import static android.content.Context.MODE_PRIVATE;

public class Utils {

    public static final String MY_PREFS_NAME = "MyPrefsFile";
    static String dirpath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/RDT/";
    static int mImageCount = 0;

    static void  CreateDirectory() {

        File file = new File(dirpath);
        if (!file.exists()) {
            file.mkdirs();
        }
    }
    static void  createDirectoryFromGivenPath(String dirPath) {

        File file = new File(dirPath);
        if (!file.exists()) {
            file.mkdirs();
        }
    }

    public static void SaveROIImage(Mat greyMat, int x1, int y1, int x2, int y2) {
        Mat tmp = new Mat();
        Imgproc.cvtColor(greyMat, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
        Imgproc.rectangle(tmp, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255), 1);
        SaveMatrix(tmp,"Grey");
        tmp.release();
    }
    public static void SavecentersImage(Mat tmp) {

        SaveMatrix(tmp,"Grey_centers");
        //tmp.release();
    }

    public static void SaveMatrix(Mat tmp,String prefix) {
        Bitmap finalBitmap = null;
        try {
            finalBitmap = getBitmapFromMat(tmp);
            saveImage(finalBitmap,prefix);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
    }
    //
    public static void SaveMatrixWithGivenPath(Mat tmp,String prefix, String path) {
        Bitmap finalBitmap = null;
        try {
            finalBitmap = getBitmapFromMat(tmp);
            saveImageNew(finalBitmap,prefix, path);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
    }

    public static Bitmap getBitmapFromMat(Mat tmp) {
        Bitmap finalBitmap;
        finalBitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(tmp, finalBitmap);
        return finalBitmap;
    }


    public static void saveImage(Bitmap m,String suff) {
        CreateDirectory();
        String s = String.format("%06d", mImageCount);
        File myImage = new File(dirpath+"Image" +suff +s+ ".jpg");
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
    //
    public static void saveImageNew(Bitmap m,String suff, String imagePath) {
        String storageLocation = Environment.getExternalStorageDirectory().getAbsolutePath()+"/RDTCROP";
        String folderPath = imagePath.substring(imagePath.indexOf("/RDT_Images/")+11,imagePath.lastIndexOf('/'));
        String imageName = imagePath.substring(imagePath.lastIndexOf('/')+1 ,imagePath.lastIndexOf(".jpg"));

        createDirectoryFromGivenPath(storageLocation+folderPath+"/");
        File myImage = new File(storageLocation+folderPath+"/"+imageName+suff+ ".jpg");
        Log.i("Saving File",myImage.getAbsolutePath());
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
    public static void saveImageInfolder(Bitmap m,String suff) {
        CreateDirectory();
        String s = String.format("%06d", mImageCount);
        File myImage = new File(dirpath+"Image" + suff +s+ ".jpg");
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
    public static Rect rotateRect(Mat in, Rect roi, int rotation){
        Rect ret= new Rect();
        if (rotation == -90)
        {
            //transpose and flip
            ret.x = roi.y;
            ret.y = roi.x;
            ret.width = roi.height;
            ret.height = roi.width;
            //now flip
            ret.x = in.height() -( roi.y+roi.height);
        }
        return ret;
    }
    public static Mat rotateFrame(Mat in, int rotation)
    {
        Mat out = in;

        if (rotation == 90)
        {
            out = in.t();
            Core.flip(out, out, 1);
        }

        else if (rotation == -90)
        {
            out = in.t();
            Core.flip(out, out, 0);

        }
        return out;
    }
    public static Short ApplySettings(Context c, RdtAPI.RdtAPIBuilder builder,RdtAPI rdt) {
        double mTopTh = ObjectDetection.mThreshold;
        //double mBotTh = ObjectDetection.mBottomThreshold;
        Short mShowImageData=0;
        Config config= new Config();
        boolean mSaveNegativeData= false;
        boolean mTrackingEnable=true;
        try {
            SharedPreferences prefs = c.getSharedPreferences(MY_PREFS_NAME, MODE_PRIVATE);

            config.mMaxScale = Short.parseShort(prefs.getString("mMaxScale",  config.mMaxScale+""));
            config.mMinScale = Short.parseShort(prefs.getString("mMinScale", config.mMinScale+""));
            config.mXMin = Short.parseShort(prefs.getString("mXMin",  config.mXMin+""));
            config.mXMax = Short.parseShort(prefs.getString("mXMax", config.mXMax+""));
            config.mYMin = Short.parseShort(prefs.getString("mYMin", config.mYMin+""));
            config.mYMax = Short.parseShort(prefs.getString("mYMax", config.mYMax+""));
            config.mMinSharpness = Float.parseFloat(prefs.getString("mMinSharpness", config.mMinSharpness +""));
            config.mMaxBrightness = Float.parseFloat(prefs.getString("mMaxBrightness", config.mMaxBrightness+""));
            config.mMinBrightness = Float.parseFloat(prefs.getString("mMinBrightness", config.mMinBrightness+""));
            config.mMaxAllowedTranslationX = Short.parseShort(prefs.getString("mMaxAllowedTranslationX", config.mMaxAllowedTranslationX+""));
            config.mMaxAllowedTranslationY = Short.parseShort(prefs.getString("mMaxAllowedTranslationY", config.mMaxAllowedTranslationY+""));
            config.mMaxFrameTranslationalMagnitude = Short.parseShort(prefs.getString("mMaxFrameTranslationalMagnitude", config.mMaxFrameTranslationalMagnitude+""));
            config.mMax10FrameTranslationalMagnitude = Short.parseShort(prefs.getString("mMax10FrameTranslationalMagnitude", config.mMax10FrameTranslationalMagnitude+""));

            mTopTh = Float.parseFloat(prefs.getString("mTopTh", mTopTh+""));
            //mBotTh = Float.parseFloat(prefs.getString("mBotTh", mBotTh+""));
            mShowImageData  = Short.parseShort(prefs.getString("mShowImageData", "0"));
            short t  = Short.parseShort(prefs.getString("mSaveNegativeData", mSaveNegativeData?"1":"0"));
            if(t!=0) mSaveNegativeData =true;
            t  = Short.parseShort(prefs.getString("mTrackingEnable", mTrackingEnable?"1":"0"));
            if(t==0) mTrackingEnable = false;

        }catch (NumberFormatException nfEx){//prefs.getString("mMinBrightness", "110.0f")
            Log.i("RDT","Exception in  Shared Pref switching to default");
            config.setDefaults();
            mShowImageData = 0;
            mSaveNegativeData = false;
        }
        if(builder != null) {
            builder.setMinBrightness(config.mMinBrightness);
            builder.setMaxBrightness(config.mMaxBrightness);
            builder.setMinScale(config.mMinScale);
            builder.setMaxScale(config.mMaxScale);
            builder.setMinSharpness(config.mMinSharpness);
            builder.setXMax(config.mXMax);
            builder.setXMin(config.mXMin);
            builder.setYMax(config.mYMax);
            builder.setYMin(config.mYMin);
            builder.setmMaxAllowedTranslationX(config.mMaxAllowedTranslationX);
            builder.setmMaxAllowedTranslationY(config.mMaxAllowedTranslationY);
        }
        if(rdt !=null){
            //rdt.getTensorFlow().setTopThreshold(mTopTh);
            rdt.setSaveNegativeData(mSaveNegativeData);
        }
        return mShowImageData;
    }

    public static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) {
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    public static int clampToMin(int x, int inp){
        return inp<0?0:inp;
    }

    public static int clampToMax(int inp,int max){
        return inp>max?max:inp;
    }

}
