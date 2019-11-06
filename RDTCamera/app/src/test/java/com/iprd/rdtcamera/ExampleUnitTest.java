package com.iprd.rdtcamera;

import org.junit.Assert;
import org.junit.Test;
import static org.hamcrest.Matchers.*;

import org.opencv.core.Mat;
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

    static Point translate(Point inp, Point t)
    {
        return new Point (inp.x+t.x,inp.y+t.y);
    }

    static void swap(float[] xy)
    {
        float t=xy[0];
        xy[0]=xy[1];
        xy[1]=t;
    }

    static Point swap(Point xy)
    {
        return new Point (xy.y,xy.x);
    }

    static void scale(float[] xy, float s)
    {
        xy[0]=xy[0]*s;
        xy[1]=xy[1]*s;;
    }
    static Point scale(Point xy, float s)
    {
        return new Point (xy.x *s,xy.y *s);
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
    @Test
    public void predictionTest2() {
        Point a = new Point(152.0f, 30.0f);
        Point c = new Point(746.0f, 30.0f);
        Point i = new Point(874.0f, 30.0f);
        float s= 0.75f;
        Point t= new Point(10,20);
        Point est_scal_rot = new Point();
        {
            Assert.assertThat("identity test",ObjectDetection.detect2(a, c, i,est_scal_rot), is(equalTo(0.0)));
            Assert.assertThat("identity_s",est_scal_rot.x,is(closeTo(1.0,0.01)));
            Assert.assertThat("identity_r",est_scal_rot.y,is(closeTo(0.0,0.01)));
        }

        {

            Point a1=translate(a,t);
            Point c1=translate(c,t);
            Point i1=translate(i,t);

            Assert.assertThat("translate test",ObjectDetection.detect2(a1, c1, i1,est_scal_rot), is(equalTo(0.0)));
            Assert.assertThat("identity_s",est_scal_rot.x,is(closeTo(1.0,0.01)));
            Assert.assertThat("identity_r",est_scal_rot.y,is(closeTo(0.0,0.01)));
        }
        {

            Point a1=scale(a,s);
            Point c1=scale(c,s);
            Point i1=scale(i,s);

            Assert.assertThat("scale test",ObjectDetection.detect2(a1, c1, i1,est_scal_rot), is(closeTo(0.0,3.0)));
            Assert.assertThat("identity_s",est_scal_rot.x,is(closeTo(0.75,0.01)));
            Assert.assertThat("identity_r",est_scal_rot.y,is(closeTo(0.0,0.01)));

        }

        {
            Point a1=swap(a);
            Point c1=swap(c);
            Point i1=swap(i);

            Assert.assertThat("rotation 90 test",ObjectDetection.detect2(a1, c1, i1,est_scal_rot), is(closeTo(0.0,3.0)));
            Assert.assertThat("identity_s",est_scal_rot.x,is(closeTo(0.0,0.01)));
            Assert.assertThat("identity_r",est_scal_rot.y,is(closeTo(Math.PI/2,0.01)));

        }

        {
            Point a1=swap(scale(translate(a,t),s));
            Point c1=swap(scale(translate(c,t),s));
            Point i1=swap(scale(translate(i,t),s));

            Assert.assertThat("rotation 90 with scale and translate test",ObjectDetection.detect2(a1, c1, i1,est_scal_rot), is(closeTo(0.0,3.0)));
            Assert.assertThat("identity_s",est_scal_rot.x,is(closeTo(0.75,0.01)));
            Assert.assertThat("identity_r",est_scal_rot.y,is(closeTo(Math.PI/2,0.01)));

        }

        {
            Mat R = ObjectDetection.makeRMat(0.85,Math.PI/6,new Point(14,23));
            Point a1=ObjectDetection.warpPoint(a,R);
            Point c1=ObjectDetection.warpPoint(c,R);
            Point i1=ObjectDetection.warpPoint(i,R);

            Assert.assertThat("affine transform test",ObjectDetection.detect2(a1, c1, i1,est_scal_rot), is(closeTo(0.0,3.0)));
            Assert.assertThat("identity_s",est_scal_rot.x,is(closeTo(0.85,0.01)));
            Assert.assertThat("identity_r",est_scal_rot.y,is(closeTo(Math.PI/6,0.01)));

        }
    }

        static {
        System.loadLibrary("opencv_java346");
    }

}