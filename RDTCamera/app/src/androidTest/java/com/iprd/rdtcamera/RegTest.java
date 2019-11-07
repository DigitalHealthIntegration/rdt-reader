package com.iprd.rdtcamera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;

import static org.junit.Assert.*;
import static org.opencv.imgproc.Imgproc.cvtColor;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class RegTest {
    byte[] ReadFile(String fname) {
        byte[] mtfliteBytes = null;
        Log.d("RDT UT","Reading file "+ fname);
        InputStream is = getClass().getResourceAsStream(fname);
        try {
            mtfliteBytes = new byte[is.available()];
            is.read(mtfliteBytes);
            is.close();
        }catch (IOException e) {
            e.printStackTrace();
            mtfliteBytes=null;
        }
        return mtfliteBytes;
    }
    Mat GetGreyMat(Bitmap inp){
        Mat out = new Mat();
        Mat grey = new Mat();
        Utils.bitmapToMat(inp, out);
        cvtColor(out, grey, Imgproc.COLOR_RGBA2GRAY);
        return grey;
    }

    private Mat getMat(String s) {
        byte[] blob = ReadFile(s);
        assertTrue("Unable to read "+s,blob !=null);
        Bitmap b =  BitmapFactory.decodeByteArray(blob, 0, blob.length);
        return GetGreyMat(b);
    }

    @Test
    public void TestRegistration() {
        Mat ins= getMat("/NotFound0.jpg");
        Mat ref= getMat("/NotFound0.jpg");

        long st  = System.currentTimeMillis();
        Mat warpmat = ImageRegistration.GetTransform(ref,ins);
        long et  = System.currentTimeMillis()-st;
        Log.d("Timing ", String.valueOf(et));

        double scale = Math.sqrt(warpmat.get(0,0)[0]*warpmat.get(0,0)[0]+warpmat.get(0,1)[0]*warpmat.get(0,1)[0]);
        Log.d("Scale ", String.valueOf(scale));

        Log.d("Madhav 0x0", String.valueOf(warpmat.get(0,0)[0]));
        Log.d("Madhav 0x1", String.valueOf(warpmat.get(0,1)[0]));
        Log.d("Madhav 1x0", String.valueOf(warpmat.get(1,0)[0]));
        Log.d("Madhav 1x1", String.valueOf(warpmat.get(1,1)[0]));

        Log.d("Madhav Tx", String.valueOf(warpmat.get(0,2)[0]));
        Log.d("Madhav Ty", String.valueOf(warpmat.get(1,2)[0]));
    }

    static {
        System.loadLibrary("opencv_java3");
    }
}
