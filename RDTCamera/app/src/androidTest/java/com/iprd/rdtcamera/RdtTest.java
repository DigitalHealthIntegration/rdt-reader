package com.iprd.rdtcamera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.IOException;
import java.io.InputStream;

import static org.junit.Assert.*;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class RdtTest {
    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();

        assertEquals("com.iprd.rdtcamera", appContext.getPackageName());
    }


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

    @Test
    public void rdtTest1() {
        RdtAPI mRdtApi;
        Config c = new Config();
        c.mTfliteB = ReadFile("/tflite.lite");
        assertTrue("Unable to read tflite file ",c.mTfliteB!=null);
        mRdtApi = new RdtAPI();
        mRdtApi.init(c);
//        mTopTh = 0.9f;
//        mBotTh = 0.7f;
//        mRdtApi.setTopThreshold(mTopTh);
//        mRdtApi.setBottomThreshold(mBotTh);
//        mRdtApi.mSaveNegativeData = mSaveNegativeData;

        byte[] blob = ReadFile("/NotFound0.jpg");
        assertTrue("Unable to read /NotFound0.jpg ",blob !=null);
        Bitmap capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);
        AcceptanceStatus status = mRdtApi.update(capFrame);
        assertTrue("RDT is found in NotFound0.jpg ",!status.mRDTFound);

        blob = ReadFile("/NotFound1.jpg");
        assertTrue("Unable to read /NotFound1.jpg ",blob !=null);
        capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);
        status = mRdtApi.update(capFrame);
        assertTrue("RDT is found in NotFound1.jpg ",!status.mRDTFound);

        blob = ReadFile("/Found0.jpg");
        assertTrue("Unable to read /Found0.jpg ",blob !=null);
        capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);
        status = mRdtApi.update(capFrame);
        assertTrue("RDT is not found in NotFound0.jpg ",status.mRDTFound);

        blob = ReadFile("/Found1.jpg");
        assertTrue("Unable to read /Found1.jpg ",blob !=null);
        capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);
        status = mRdtApi.update(capFrame);
        assertTrue("RDT is not found in Found1.jpg ",status.mRDTFound);

    }
    static {
        System.loadLibrary("opencv_java3");
    }
}
