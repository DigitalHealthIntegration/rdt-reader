package com.iprd.testapplication;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.util.Log;

import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgproc;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.opencv.core.CvException;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

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
        cvtColor(greyMat, tmp, opencv_imgproc.COLOR_GRAY2RGBA, 4);
        Scalar s = new Scalar(0,0,255,0);
        opencv_imgproc.rectangle(tmp, new Point(x1, y1), new Point(x2, y2), s);
        Bitmap finalBitmap = null;
        try {
            //Imgproc.cvtColor(seedsImage, tmp, Imgproc.COLOR_RGB2BGRA);
            finalBitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);

            AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
            OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
            finalBitmap = converterToBitmap.convert(converterToMat.convert(tmp));
            //opencv_imgproc.matToBitmap(tmp, finalBitmap);
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

    public static Short ApplySettings(Context c, RdtAPI.RdtAPIBuilder builder,RdtAPI rdt) {
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
            mTopTh = Float.parseFloat(prefs.getString("mTopTh", mTopTh+""));
            mBotTh = Float.parseFloat(prefs.getString("mBotTh", mBotTh+""));
            mShowImageData  = Short.parseShort(prefs.getString("mShowImageData", "0"));
            short t  = Short.parseShort(prefs.getString("mSaveNegativeData", mSaveNegativeData?"1":"0"));
            if(t!=0) mSaveNegativeData =true;
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
        }
        if(rdt !=null){
            rdt.getTensorFlow().setBottomThreshold(mBotTh);
            rdt.getTensorFlow().setTopThreshold(mTopTh);
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

}
