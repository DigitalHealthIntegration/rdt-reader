package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.junit.Assert;
import org.junit.Test;
import static org.hamcrest.Matchers.*;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.cvtColor;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleInstrumentedTest {
    byte[] ReadFile(String fname) {
        byte[] mtfliteBytes = null;
        //Log.d("RDT UT","Reading file "+ fname);
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
        assertTrue("Unable toread "+s,blob !=null);
        Bitmap b =  BitmapFactory.decodeByteArray(blob, 0, blob.length);
        return GetGreyMat(b);
    }


    @Test
    public void TestSetNV12() {
        int width=928;
        int height = 720;
        int roix=25;
        int roiy=100;
        int roiwidth=20;
        int roiheight = 80;

        int[] img;
        img = new int[(int)(width*height*1.5)];
        for(int i=0;i< roiheight;i++){
            int offsetY = roiy*width;
            int offsetUV = width*height + (roiy>>1)*width;

            for(int j=0;j<roiwidth;j++){
                int offsetYTotal = offsetY+roix+j;
                int offsetUVTotal = offsetUV+ 2*((roix+j)>>1);
                //lets write first and last row
                if((i ==0) ||(i ==(roiheight -1))){
                    img[offsetYTotal]=255;
                    img[offsetUVTotal]=0;
                    img[offsetUVTotal+1]=0;
                }else if((j==0)||(j == (roiwidth-1))){
                    img[offsetYTotal]=255;
                    img[offsetUVTotal]=0;
                    img[offsetUVTotal+1]=0;
                }
            }
        }
    }

        @Test
    public void TestRegistration() {
        try {
            //Mat ins = getMat("/NotFound0.jpg");
            //Mat ref = getMat("/NotFound0.jpg");
            byte data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        Mat ref =  new Mat(3, 3, CV_8U, ByteBuffer.wrap(data));
        Mat ins =  new Mat(3, 3, CV_8U, ByteBuffer.wrap(data));

            long st = System.currentTimeMillis();
            Mat warpmat = ImageRegistration.getTransformation(ref, ins);
            long et = System.currentTimeMillis() - st;
            //       Log.d("Timing ", String.valueOf(et));

            double scale = Math.sqrt(warpmat.get(0, 0)[0] * warpmat.get(0, 0)[0] + warpmat.get(0, 1)[0] * warpmat.get(0, 1)[0]);

            Assert.assertThat("Scale Tx test", scale, is(equalTo(1.0)));
            Assert.assertThat("Tx test", warpmat.get(0, 2)[0], is(equalTo(0.0)));
            Assert.assertThat("Ty test", warpmat.get(1, 2)[0], is(equalTo(0.0)));

            Assert.assertThat("rx test", warpmat.get(0, 0)[0], is(equalTo(0.0)));
            Assert.assertThat("ry test", warpmat.get(1, 1)[0], is(equalTo(0.0)));

        }
        catch (Exception ex){
            ex.printStackTrace();
        }
//

//        Log.d("Scale ", String.valueOf(scale));
//
//        Log.d("Madhav 0x0", String.valueOf(warpmat.get(0,0)[0]));
//        Log.d("Madhav 0x1", String.valueOf(warpmat.get(0,1)[0]));
//        Log.d("Madhav 1x0", String.valueOf(warpmat.get(1,0)[0]));
//        Log.d("Madhav 1x1", String.valueOf(warpmat.get(1,1)[0]));
//
//        Log.d("Madhav Tx", String.valueOf(warpmat.get(0,2)[0]));
//        Log.d("Madhav Ty", String.valueOf(warpmat.get(1,2)[0]));
    }

    static {
        System.loadLibrary("opencv_java346");
    }
}
