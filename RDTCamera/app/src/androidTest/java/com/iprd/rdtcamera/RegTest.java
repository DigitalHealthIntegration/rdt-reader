package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import static com.iprd.rdtcamera.CvUtils.ComputeVector;
import static com.iprd.rdtcamera.CvUtils.PrintAffineMat;
import static com.iprd.rdtcamera.CvUtils.mComputeVector_FinalPoint;
import static com.iprd.rdtcamera.CvUtils.scaleAffineMat;
import static com.iprd.rdtcamera.ImageRegistration.ComputeMotion;
import static com.iprd.rdtcamera.ObjectDetection.warpPoint;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.*;
import static org.opencv.core.CvType.CV_8U;
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
        if(out.type()!=CV_8U){
            cvtColor(out, grey, Imgproc.COLOR_RGBA2GRAY);
        }else{
            Mat BGRImage = new Mat (inp.getWidth(), inp.getHeight(), CV_8U);
            Utils.bitmapToMat(inp, BGRImage);
            grey = BGRImage.clone();
        }
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

    @Test
    public void TestRegWithROI() {
        Mat ref=null;
        Mat a = getMap("/NotFound0.jpg");
        Mat warpmat = ImageRegistration.FindMotion(a,true);
//        Assert.assertThat("Null WarpMat",warpmat,null);
        Mat b = getMap("/NotFound0.jpg");
        warpmat = ImageRegistration.FindMotion(b,true);
        assertEquals("Scale Tx test",warpmat.get(0,0)[0],1.0,0.001);
        assertEquals("Scale Ty test",warpmat.get(1,1)[0],1.0,0.001);

        ///Now say RDT is busy and we try track it.
        Point lt= new Point(164,275);
        Point rb= new Point(290,1804);

        Mat c = getMap("/NotFound1.jpg");
        Mat warpmat1 = ImageRegistration.FindMotion(c,true);

        Point ltd = warpPoint(new Point(lt.x/16.0,lt.y/16.0),warpmat1);
        ltd.x = ltd.x*16;
        ltd.y = ltd.y*16;

        Mat warp= warpmat1.clone();
        //level 4 mat
        int factor = 1<<4;
        warp.put(0,1,warp.get(0,1)[0]*factor);
        warp.put(1,0,warp.get(1,0)[0]*factor);
        warp.put(0,2,warp.get(0,2)[0]*factor);
        warp.put(1,2,warp.get(1,2)[0]*factor);


        Point lt1 = warpPoint(lt,warp);
        Point rb1 = warpPoint(rb,warp);


        Mat warpmat2 = ImageRegistration.FindMotion(c,true);
        Point lt2 = warpPoint(lt1,warpmat2);
        Point rb2 = warpPoint(rb1,warpmat2);
        assertEquals("LT X should be same  ",lt1.x ,lt2.x ,0.001);
        assertEquals("LT Y should be same  ",lt1.y ,lt2.y ,0.001);
        assertEquals("RB X should be same  ",rb1.x ,rb2.x ,0.001);
        assertEquals("RB Y should be same  ",rb1.y ,rb2.y ,0.001);
    }

    @Test
    public void TestReg() {
        long st  = System.currentTimeMillis();
        Mat a = getMap("/NotFound0.jpg");
        Mat b = getMap("/NotFound1.jpg");
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

    @Test
    public void TestComputeMotion() {
        long st  = System.currentTimeMillis();
        String prefname = "/Tx/Image";
        String postfname = "Input.jpg";
        String name = prefname+"13"+postfname;
        Mat input = getMap(name);
        ArrayList<Mat> arrMat= new ArrayList<>();
        Rect init=new Rect(685,199,72,1276);
        Mat warp = ComputeMotion(input);
        for(int i=14;i<31;i++) {
            name = prefname + i + postfname;
            input = getMap(name);
            warp = ComputeMotion(input);
            arrMat.add(warp);
            PrintAffineMat(""+i,warp);
            Point p = null;
            Scalar s = new Scalar(255,0,0,255);
            Mat mag = ComputeVector(new Point(warp.get(0,2)[0],warp.get(0,2)[0]),null,s);
            Log.i("Point ", mComputeVector_FinalPoint.x+"x"+mComputeVector_FinalPoint.y);
            Point lt = CvUtils.warpPoint(new Point(init.x, init.y), warp);
            Point rb = CvUtils.warpPoint(new Point(init.x + init.width, init.y + init.height), warp);
            Imgproc.circle(input, lt, 3, new Scalar(0, 255, 0,0), 2);
            Imgproc.circle(input, rb, 3, new Scalar(0, 255, 0,0), 2);
            com.iprd.rdtcamera.Utils.SaveMatrix(input,"Test");
            init.x = (int) lt.x;
            init.y = (int) lt.y;
            init.width = (int) (rb.x-lt.x);
            init.height= (int) (rb.y-lt.y);
            assertTrue("Tx should be -ve ",warp.get(0,2)[0]< 0);

        }
    }

    static {
        System.loadLibrary("opencv_java3");
    }
}
