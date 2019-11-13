package com.iprd.rdtcamera;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Environment;
import android.util.Log;


import androidx.test.ext.junit.runners.AndroidJUnit4;

//import androidx.test.runner.AndroidJUnit4;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
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
        String imgPath = Environment.getExternalStorageDirectory().getPath()+"/RDT_Images";
        ArrayList<String> list = getImageList(imgPath);
        for (String s: list
             ) {
            byte[] blob = ReadFromSdcard(s);
            Log.i("Size ",s+"->"+blob.length);
            assertTrue("Unable to read " + imgPath,blob !=null);
        }
    }

    @Test
    public void test2(){
        RdtAPI mRdtApi=getRdtAPI();

        String imgPath = Environment.getExternalStorageDirectory().getPath()+"/RDT_Images";
        ArrayList<String> list = getImageList(imgPath);
        for (String s: list
        ) {
            byte[] blob = ReadFromSdcard(s);
           // Log.i("File Info ", s + "->" + blob.length);
            assertTrue("Unable to read " + imgPath, blob != null);
            Bitmap capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);
            AcceptanceStatus status = mRdtApi.checkFrame(capFrame);
            Log.i("Result ", s + " : " + status.GetResult());
        }
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
        mRdtApi.setLinearflow(true);
        return mRdtApi;
    }

    byte[] ReadFromSdcard(String s)  {
        File file = new File(s);
        int length = (int) file.length();
        byte[] bytes = null;
        FileInputStream in = null;
        try {
            in = new FileInputStream(file);
            bytes = new byte[length];
            in.read(bytes);
            in.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
        }
        return bytes;
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
        InputStream is=this.getClass().getClassLoader().getResourceAsStream("OD_180x320.lite");//OD_180x320.lite");//"OD_360x640_10x19_slow.lite");
        mtfliteBytes=new byte[is.available()];
        is.read( mtfliteBytes);
        is.close();
        return mtfliteBytes;
    }

    private static ArrayList<String> getImageList(String folderPath) {
        ArrayList<String> ar = new ArrayList<>();
        //folderPath = "/home/developer/RDTCLONE/rdt-reader/RDTCamera/app/sampledata/FluAB";
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
