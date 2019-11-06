package com.iprd.rdtcamera;

import org.junit.Assert;
import org.junit.Test;
import static org.hamcrest.CoreMatchers.*;
import org.opencv.core.Point;
import org.opencv.core.Size;

import java.io.IOException;
import java.io.InputStream;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    byte[] mtfliteBytes = null;

    byte[] ReadAssests() throws IOException {
        InputStream is=this.getClass().getClassLoader().getResourceAsStream("tflite.lite");
        //InputStream in = this.getClass().getClassLoader().getResourceAsStream("myFile.txt");
        mtfliteBytes=new byte[is.available()];
        is.read( mtfliteBytes);
        is.close();
        return mtfliteBytes;
    }

    @Test
    public void rdtTest1() {
        RdtAPI mRdtApi;


        Config c = new Config();
        try {
            c.mTfliteB = ReadAssests();
        } catch (IOException e) {
            e.printStackTrace();
        }
/*
        mRdtApi = new RdtAPI();
        mRdtApi.init(c);
*/

        assertEquals(4, 2 + 2);
    }

    static void swap(float[] xy)
    {
        float t=xy[0];
        xy[0]=xy[1];
        xy[1]=t;
    }
    static void scale(float[] xy, float s)
    {
        xy[0]=xy[0]*s;
        xy[1]=xy[1]*s;;
    }
    @Test
    public void predictionTest() {
        float[] Arrow = {152.0f, 30.0f};
        float[] Cpattern = {746.0f, 30.0f};
        float[] Infl = {874.0f, 30.0f};

        {//

            Assert.assertThat("identity test",ObjectDetection.detect(Arrow, Cpattern, Infl), is(equalTo(0.0)));
        }
        {//
            Arrow[1] += 100;
            Cpattern[1] += 100;
            Infl[1] += 100;
            Assert.assertThat("translation test",ObjectDetection.detect(Arrow, Cpattern, Infl), is(equalTo(0.0)));
        }

        {//
            float s=0.75f;
            scale(Arrow,s);
            scale(Cpattern,s);
            scale(Infl,s);
            Assert.assertThat("scale test",ObjectDetection.detect(Arrow, Cpattern, Infl), is(equalTo(0.0)));
        }

        {//90 rotation
            swap(Arrow);
            swap(Cpattern);
            swap(Infl);
            Assert.assertThat("rotation test",ObjectDetection.detect(Arrow, Cpattern, Infl), is(equalTo(0.0)));
        }


    }

        static {
        System.loadLibrary("opencv_java346");
    }

}