package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.*;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.pyrDown;
import static org.opencv.video.Video.MOTION_EUCLIDEAN;
import static org.opencv.video.Video.findTransformECC;
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

    private Bitmap getBitmap(String s) {
        byte[] blob = ReadFile(s);
        assertTrue("Unable to read "+s,blob !=null);
        Bitmap b =  BitmapFactory.decodeByteArray(blob, 0, blob.length);
        return b;//GetGreyMat(b);
    }
    private Mat getMap(String s) {
        byte[] blob = ReadFile(s);
        assertTrue("Unable to read "+s,blob !=null);
        Bitmap b =  BitmapFactory.decodeByteArray(blob, 0, blob.length);
        return GetGreyMat(b);
    }

    public static Bitmap alignImagesEuclidean(Bitmap A, Bitmap B)
    {
        final int warp_mode = MOTION_EUCLIDEAN;
        Mat matA = new Mat(A.getHeight(), A.getWidth(), CvType.CV_8UC3);
        Mat matAgray = new Mat(A.getHeight(), A.getWidth(), CvType.CV_8U);
        Mat matB = new Mat(B.getHeight(), B.getWidth(), CvType.CV_8UC3);
        Mat matBgray = new Mat(B.getHeight(), B.getWidth(), CvType.CV_8U);
        Mat matBaligned = new Mat(A.getHeight(), A.getWidth(), CvType.CV_8UC3);
        Mat warpMatrix = Mat.eye(2,3,CV_32F);
        Utils.bitmapToMat(A, matA);
        Utils.bitmapToMat(B, matB);
        Imgproc.cvtColor(matA, matAgray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(matB, matBgray, Imgproc.COLOR_BGR2GRAY);

        //Mat warpmat = ImageRegistration.GetTransform(matAgray,matBgray);


        int numIter = 5;
        double terminationEps = 1e-10;
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, numIter, terminationEps);
        findTransformECC(matAgray, matBgray, warpMatrix, warp_mode, criteria, matBgray);
        Imgproc.warpAffine(matB, matBaligned, warpMatrix, matA.size(), Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP);
        Bitmap alignedBMP = Bitmap.createBitmap(A.getWidth(), A.getHeight(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(matBaligned, alignedBMP);
        return alignedBMP;
    }

    @Test
    public void TestReg() {
        Bitmap ins= getBitmap("/NotFound0.jpg");
        Bitmap ref= getBitmap("/NotFound1.jpg");

        long st  = System.currentTimeMillis();
        alignImagesEuclidean(ins,ref);

        Mat a = getMap("/NotFound0.jpg");
        Mat b = getMap("/NotFound0.jpg");
        Mat a1 = new Mat();
        Mat b1 = new Mat();
        pyrDown(a, a1);
        pyrDown(b, b1);
        Mat warpmat = ImageRegistration.GetTransform(a1,b1);
        long et  = System.currentTimeMillis()-st;
        Log.i("Timing ", String.valueOf(et));

        double scale = Math.sqrt(warpmat.get(0,0)[0]*warpmat.get(0,0)[0]+warpmat.get(0,1)[0]*warpmat.get(0,1)[0]);
        Log.i("Scale ", String.valueOf(scale));
        Log.i("Madhav 0x0", String.valueOf(warpmat.get(0,0)[0]));
        Log.i("Madhav 0x1", String.valueOf(warpmat.get(0,1)[0]));
        Log.i("Madhav 1x0", String.valueOf(warpmat.get(1,0)[0]));
        Log.i("Madhav 1x1", String.valueOf(warpmat.get(1,1)[0]));

        Log.i("Madhav Tx", String.valueOf(warpmat.get(0,2)[0]));
        Log.i("Madhav Ty", String.valueOf(warpmat.get(1,2)[0]));

        Assert.assertThat("Scale Tx test",scale, is(equalTo(1.0)));
//        Assert.assertThat("Tx test",warpmat.get(0,2)[0], is(equalTo(0.0001)));
//        Assert.assertThat("Ty test",warpmat.get(1,2)[0], is(equalTo(0.0001)));
//
//        Assert.assertThat("rx test",warpmat.get(0,0)[0], is(equalTo(0.0001)));
//        Assert.assertThat("ry test",warpmat.get(1,1)[0], is(equalTo(0.0001)));
    }
    static {
        System.loadLibrary("opencv_java3");
    }
}
