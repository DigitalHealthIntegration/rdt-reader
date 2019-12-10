package com.iprd.rdtcamera;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Environment;
import android.util.Log;


import androidx.test.ext.junit.runners.AndroidJUnit4;

//import androidx.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.core.Mat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import org.opencv.android.Utils;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import static com.iprd.rdtcamera.Utils.SaveMatrix;
import static com.iprd.rdtcamera.Utils.SaveMatrixWithGivenPath;
import static com.iprd.rdtcamera.Utils.getBitmapFromMat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.opencv.core.Core.LINE_AA;
import static org.opencv.imgproc.Imgproc.putText;

@RunWith(AndroidJUnit4.class)
public class RdtFrameTest {

    @Test
    public void test1(){
//        String imgPath = Environment.getExternalStorageDirectory().getPath()+"/RDT_Images";
//        ArrayList<String> list = getImageList(imgPath);
//        for (String s: list) {
//            byte[] blob = ReadFromSdcard(s);
//            Log.i("Size ",s+"->"+blob.length);
//            assertTrue("Unable to read " + imgPath,blob !=null);
//        }
    }

    @Test
    public void test2(){
        RdtAPI mRdtApi=getRdtAPI();

        String imgPath = Environment.getExternalStorageDirectory().getPath()+"/RDT_Images";
        ArrayList<String> list = getImageList(imgPath);
        FileWriter fileWriter;
        BufferedWriter bfWriter = null;

        if(list.size() > 0) {
            String fileDir = Environment.getExternalStorageDirectory().getAbsolutePath() + "/RDT_Images_Folder/";
            File f = new File(fileDir);
            if(!f.exists()){
                f.mkdir();
            }
            File file = new CreateCsvFile(fileDir, "RDT_Images_Results.csv").invoke();
            for (int i=0; i<list.size(); i++) {
                //String s = "/storage/emulated/0/RDT_Images/FluA/Morning/IMG_1613.jpg";
                byte[] blob = ReadFromSdcard(list.get(i));
                // Log.i("File Info ", s + "->" + blob.length);
                assertTrue("Unable to read " + imgPath, blob != null);
                Bitmap capFrame = BitmapFactory.decodeByteArray(blob, 0, blob.length);
                Mat matinput = new Mat();
                Utils.bitmapToMat(capFrame, matinput);
                if(matinput.width()< matinput.height()){
                    //Rotate mat;
                    matinput = com.iprd.rdtcamera.Utils.rotateFrame(matinput, -90);
                    //Find the biggest rectangle.
                }
                //Lets Rescale it.

                int width = matinput.width();
                int height = matinput.height();
                int wfactor = width/16;
                int hfactor = height/9;
                int factor  = wfactor;//wfactor>hfactor?hfactor:wfactor;
                int newWidth = factor*16;
                int newHeight = factor*9;
                Rect r = new Rect((width - newWidth)/2,(height-newHeight)/2,newWidth,newHeight);
                Mat croppedImage = matinput.submat(r);
                capFrame = getBitmapFromMat(croppedImage);
//                SaveMatrixWithGivenPath(croppedImage,"",list.get(i));
                AcceptanceStatus status = mRdtApi.checkFrame(capFrame);

                //Bitmap intermediate = getBitmapFromMat(mRdtApi.getTensorFlow().tmp_for_draw);
                Mat m = mRdtApi.getTensorFlow().tmp_for_draw;
                if(mRdtApi.getTensorFlow().tmp_for_draw!= null) SaveMatrixWithGivenPath(mRdtApi.getTensorFlow().tmp_for_draw,"_I",list.get(i));

                Log.i("Result ", list.get(i) + " : " + status.GetResult());
                if (file.exists()) {
                    try {
                        if(i == 0) {
                            fileWriter = new FileWriter(file);
                            bfWriter = new BufferedWriter(fileWriter);
                            bfWriter.write("Image, mSharpValue, mScale,mAngle, mBrightValue, mRDTFound, mBoundingBoxX, mBoundingBoxY, mBoundingBoxWidth,mBoundingBoxHeight,mError\n");
                        }
                        bfWriter.write(list.get(i)+","+ status.mInfo.mSharpness+","+ status.mInfo.mScale+","+status.mInfo.mAngle+","+ status.mInfo.mBrightness+","+ status.mRDTFound
                                    +","+ status.mBoundingBoxX+","+ status.mBoundingBoxY+","+ status.mBoundingBoxWidth+","+status.mBoundingBoxHeight+","+status.mInfo.mMinError+"\n");
                        if(i == list.size()-1) {
                            bfWriter.close();
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
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
        mRdtApi.setLinearflow(true);
        mRdtApi.setmPlaybackMode(true);
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
        InputStream is=this.getClass().getClassLoader().getResourceAsStream("OD_180x320_newarch.lite");//OD_180x320.lite");//"OD_360x640_10x19_slow.lite");
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

    private class CreateCsvFile {
        private String fileDir;
        private String fileName;

        public CreateCsvFile(String fileDir, String fileName) {
            this.fileDir = fileDir;
            this.fileName = fileName;
        }

        public File invoke() {
            File file = new File(fileDir+File.separator+fileName);

            if(!file.exists()) {
                try {file.createNewFile();}
                catch (IOException e) {e.printStackTrace();}
            }
            return file;
        }
    }
}
