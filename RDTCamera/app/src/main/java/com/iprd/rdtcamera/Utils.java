package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.os.Environment;
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

    void CreateDirectory() {

        File file = new File(dirpath);
        if (!file.exists()) {
            file.mkdirs();
        }
    }

    public void SaveROIImage(Mat greyMat, int x1, int y1, int x2, int y2) {
        Mat tmp = new Mat();
        Imgproc.cvtColor(greyMat, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
        Imgproc.rectangle(tmp, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255), 1);
        Bitmap finalBitmap = null;
        try {
            //Imgproc.cvtColor(seedsImage, tmp, Imgproc.COLOR_RGB2BGRA);
            finalBitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(tmp, finalBitmap);
            saveImage(finalBitmap);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        tmp.release();
    }

    public void saveImage(Bitmap m) {
        File myImage = new File(dirpath+"Image" + mImageCount + ".jpg");
        mImageCount++;
        if (myImage.exists()) myImage.delete();
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(myImage);
            m.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
