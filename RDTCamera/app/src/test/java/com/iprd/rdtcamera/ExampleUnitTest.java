package com.iprd.rdtcamera;

import org.junit.Test;

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
        Config c = new Config();
        try {
            c.mTfliteB = ReadAssests();
        } catch (IOException e) {
            e.printStackTrace();
        }
        RdtAPI mRdtApi;
        RdtAPI.RdtAPIBuilder builder;
        builder = new RdtAPI.RdtAPIBuilder();
        builder = builder.setByteModel(c.mTfliteB);

        mRdtApi = builder.build();


        assertEquals(4, 2 + 2);
    }
}