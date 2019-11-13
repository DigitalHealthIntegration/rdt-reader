package com.iprd.rdtcamera;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.util.Log;


import androidx.test.ext.junit.runners.AndroidJUnit4;

//import androidx.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class RdtFrameTest {

    @Test
    public void test1(){
        assertEquals("Mono","Mono");
    }

    @Test
    public void test2(){
        RdtAPI mRdtApi=getRdtAPI();

        String imgPath = "/IMG_1564.jpg";//"/home/developer/Documents/RdtReader/rdt-reader/RDTTestImage/Negative/Morning/IMG_1564.jpg";//
        byte[] blob = ReadFile(imgPath);
        assertTrue("Unable to read " + imgPath,blob !=null);
        Bitmap capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);

        AcceptanceStatus status = mRdtApi.checkFrame(capFrame);

        assertTrue("RDT is found in "+imgPath,!status.mRDTFound);

    }


    public RdtAPI getRdtAPI(){
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
        mRdtApi.setSavePoints(true);
        mRdtApi.setSaveImages(true);
        mRdtApi.setTracking(false);
        return mRdtApi;
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

    byte[] mtfliteBytes = null;
    byte[] ReadAssests() throws IOException {
        InputStream is=this.getClass().getClassLoader().getResourceAsStream("OD_360x640_10x19_slow.lite");//OD_180x320.lite");//"OD_360x640_10x19_slow.lite");
        mtfliteBytes=new byte[is.available()];
        is.read( mtfliteBytes);
        is.close();
        return mtfliteBytes;
    }

    private static ArrayList<String> getImageList(String folderPath) {
        ArrayList<String> ar = new ArrayList<>();
        folderPath = "/home/developer/RDTCLONE/rdt-reader/RDTCamera/app/sampledata/FluAB";
        try {
             if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Files.walk(Paths.get(folderPath))
                    .filter(path -> Files.isRegularFile(path))
                    .forEach(a->ar.add(a.toString()));
            }else{
                System.out.println("Min SDK should be 26");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(ar);
        return ar;
    }
    static {
        System.loadLibrary("opencv_java3");
    }
}
