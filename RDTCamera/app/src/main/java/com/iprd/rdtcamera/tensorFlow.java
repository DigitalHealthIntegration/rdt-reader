package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import static org.opencv.imgproc.Imgproc.cvtColor;

public class tensorFlow {
    Interpreter mTflite;
    static int mImageCount = 0;

    tensorFlow(){
        File modelFile = new File("/mnt/sdcard/mgd/tflite.lite");
        mTflite = new Interpreter(modelFile);
    }

    class ScoreComparator implements Comparator<HashMap<Float,Vector<Integer>>> {
        public int compare(HashMap<Float,Vector<Integer>> s1,HashMap<Float,Vector<Integer>> s2){

            Map.Entry<Float,Vector<Integer>> entryS1 = s1.entrySet().iterator().next();
            Float keyS1 = entryS1.getKey();

            Map.Entry<Float,Vector<Integer>> entryS2 = s2.entrySet().iterator().next();
            Float keyS2 = entryS2.getKey();
            return Float.compare(keyS2, keyS1);
        }
    }
    int[] cxcy2xy(float cx,float cy,float w,float h){
        int [] xy =new int[4];
        xy[0]= (int) (cx-w/2);
        xy[1]= (int) (cy-h/2);
        xy[2]= (int) (cx+w/2);
        xy[3]= (int) (cy+h/2);
        return xy;
    }

    public int[] nonMaxSupression(ArrayList<HashMap<Float, Vector<Integer>>> Top,ArrayList<HashMap<Float, Vector<Integer>>>  Bottom){
        int topn=Math.min(Top.size(),Bottom.size());
        Log.d("detection","done");
        int [] roi = new int[4];
        int [] x1y1x2y2Top = {256,256,0,0};
        int [] x1y1x2y2Bot= {256,256,0,0};
        for (int i =0;i<topn;i++){
            try {
                for (Map.Entry mapElement : Top.get(i).entrySet()) {
                    float key = (float) mapElement.getKey();

                    // Add some bonus marks
                    // to all the students and print it
                    Vector x1y1x2y2 = (Vector) mapElement.getValue();
                    if(x1y1x2y2Top[0]>(int) x1y1x2y2.get(0)){
                        x1y1x2y2Top[0] = (int) x1y1x2y2.get(0);
                    }
                    if(x1y1x2y2Top[1]>(int) x1y1x2y2.get(1)){
                        x1y1x2y2Top[1] = (int) x1y1x2y2.get(1);
                    }
                    if(x1y1x2y2Top[2]<(int) x1y1x2y2.get(2)){
                        x1y1x2y2Top[2] = (int) x1y1x2y2.get(2);
                    }
                    if(x1y1x2y2Top[3]<(int) x1y1x2y2.get(3)){
                        x1y1x2y2Top[3] = (int) x1y1x2y2.get(3);
                    }

                }
                for (Map.Entry mapElement : Bottom.get(i).entrySet()) {
                    float key = (float) mapElement.getKey();

                    // Add some bonus marks
                    // to all the students and print it
                    Vector x1y1x2y2 = (Vector) mapElement.getValue();
                    if(x1y1x2y2Bot[0]>(int) x1y1x2y2.get(0)){
                        x1y1x2y2Bot[0] = (int) x1y1x2y2.get(0);
                    }
                    if(x1y1x2y2Bot[1]>(int) x1y1x2y2.get(1)){
                        x1y1x2y2Bot[1] = (int) x1y1x2y2.get(1);
                    }
                    if(x1y1x2y2Bot[2]<(int) x1y1x2y2.get(2)){
                        x1y1x2y2Bot[2] = (int) x1y1x2y2.get(2);
                    }
                    if(x1y1x2y2Bot[3]<(int) x1y1x2y2.get(3)){
                        x1y1x2y2Bot[3] = (int) x1y1x2y2.get(3);
                    }
                }

            }catch (IndexOutOfBoundsException e){
                e.printStackTrace();
            }
        }

        roi[0]=(int) Math.min(x1y1x2y2Bot[0],x1y1x2y2Top[0]);
        roi[1]= (int) Math.min(x1y1x2y2Bot[1],x1y1x2y2Top[1]);
        roi[2]= (int) Math.max(x1y1x2y2Bot[2],x1y1x2y2Top[2]);
        roi[3]= (int) Math.max(x1y1x2y2Bot[3],x1y1x2y2Top[3]);
//        roi[2] = roi[2]-roi[0];
//        roi[3] = roi[3]-roi[1];
        return roi;
    }

    Rect update(Mat inputmat,Boolean [] rdt){
        int [] roi = new int[4];
        int width = inputmat.cols();
        int height = inputmat.rows();
        Rect ret = new Rect(-1,-1,-1,-1);
        //Resize image to 256x256 for the neural network
        Mat greyMat = new Mat();
        org.opencv.core.Size sz = new org.opencv.core.Size(256, 256);
        Imgproc.resize(inputmat, greyMat, sz);

        //Feed image pixels in normalized form to the input
        float[][][][] input = new float[1][256][256][1];
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                double[] pixelvalue = greyMat.get(i, j);
                // Log.d("val"+i+"x"+j, String.valueOf(pixelvalue[0]));
//                    double normalized = (double) (greyMat.get(i,j)/255.0);
                input[0][i][j][0] = (float) (pixelvalue[0] / 255.0);
            }
        }
        //Initialize output buffer
        float[][][][] output = new float[1][225][5][7];
        //Image to draw roi in
        Bitmap bmp = null;

        long startTime = System.nanoTime();

        mTflite.run(input, output);
        long endTime = System.nanoTime();
        long MethodeDuration = (endTime - startTime);
        float timetaken = MethodeDuration/1000000.0f;
        Log.d("mTfliteTime", String.valueOf(timetaken));

//            AcceptanceStatus ret = update(mat.getNativeObjAddr());
//            if(null != ret ) Log.d("RDT FOUND ",ret.mRDTFound?"1":"0");
//            Log.d("TF","done");
        int[] anchors = new int[]{20, 10, 10, 20, 30, 30, 25, 20, 20, 25};
//           [[[20,10],[10,20],[30,30],[25,20],[20,25]]]

        ArrayList<HashMap<Float, Vector<Integer>>> vectorTableBottom = new ArrayList<HashMap<Float, Vector<Integer>>>();
        ArrayList<HashMap<Float, Vector<Integer>>> vectorTableTop= new ArrayList<HashMap<Float, Vector<Integer>>>();

        float resizeFactor = 256.0f / 15.0f;
        for (int row = 0; row < 15; row++) {
            for (int col = 0; col < 15; col++) {
                for (int j = 0; j < 5; j++) {

                    float conf_0 = output[0][row * 15 + col][j][0];
                    float conf_1 = output[0][row * 15 + col][j][1];
                    if (conf_0 > 0.8) {
//                            Log.d("outpu", String.valueOf(conf_0));
                        float cy = (float) ((row + 0.5) * resizeFactor + output[0][row * 15 + col][j][3] * 256);
                        float cx = (float) ((col + 0.5) * resizeFactor + output[0][row * 15 + col][j][4] * 256);
                        float w = anchors[j * 2] + output[0][row * 15 + col][j][5] * 256;
                        float h = anchors[j * 2 + 1] + output[0][row * 15 + col][j][6] * 256;
                        Vector v = new Vector();

                        int[] xy = cxcy2xy(cx, cy, w, h);
                        v.add(xy[0]);
                        v.add(xy[1]);
                        v.add(xy[2]);
                        v.add(xy[3]);
                        HashMap<Float, Vector<Integer>> hMap = new HashMap<Float, Vector<Integer>>();
                        hMap.put(conf_0, v);
                        vectorTableTop.add(hMap);
                    } else if (conf_1 > 0.8) {
//                            Log.d("outpu", String.valueOf(conf_1));
                        float cy = (float) ((row + 0.5) * resizeFactor + output[0][row * 15 + col][j][3] * 256);
                        float cx = (float) ((col + 0.5) * resizeFactor + output[0][row * 15 + col][j][4] * 256);

                        float w = anchors[j * 2] + output[0][row * 15 + col][j][5] * 256;
                        float h = anchors[j * 2 + 1] + output[0][row * 15 + col][j][6] * 256;

                        Vector v = new Vector();
                        int[] xy = cxcy2xy(cx, cy, w, h);
                        v.add(xy[0]);
                        v.add(xy[1]);
                        v.add(xy[2]);
                        v.add(xy[3]);
                        HashMap<Float, Vector<Integer>> hMap = new HashMap<Float, Vector<Integer>>();
                        hMap.put(conf_1, v);
                        vectorTableBottom.add(hMap);
                    }
                }
            }
        }

        if (vectorTableBottom.size() > 0) {
            Collections.sort(vectorTableBottom, new ScoreComparator());
        }
        if (vectorTableTop.size() > 0) {
            Collections.sort(vectorTableTop, new ScoreComparator());
        }
        if(vectorTableBottom.size() > 0 & vectorTableTop.size() > 0){

            roi=nonMaxSupression(vectorTableTop,vectorTableBottom);
            int x1 = roi[0];
            int y1=roi[1];
            int x2=roi[2];
            int y2=roi[3];
            //SaveROIImage(greyMat, x1, y1, x2, y2);
            ret.x = (int) (roi[0]/256.0f*width);
            ret.y = (int) (roi[1]/256.0f*height);
            ret.width = (int) (roi[2]/256.0f*width);
            ret.height = (int) (roi[3]/256.0f*height);
            rdt[0]=true;
            SaveROIImage(inputmat, ret.x,ret.y,ret.width,ret.height);
        }
        return ret;
    }

    public void SaveROIImage(Mat greyMat, int x1, int y1, int x2, int y2) {
        Mat tmp = new Mat();
        Imgproc.cvtColor(greyMat, tmp, Imgproc.COLOR_GRAY2RGBA, 4);
        Imgproc.rectangle(tmp, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255), 1);
        Bitmap finalBitmap = null;
        try {
            //Imgproc.cvtColor(seedsImage, tmp, Imgproc.COLOR_RGB2BGRA);
            finalBitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(tmp, finalBitmap);
            saveImage(finalBitmap);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        tmp.release();
    }

    public void saveImage(Bitmap m) {
        File myImage = new File("/mnt/sdcard/mgd/Image" + mImageCount + ".jpg");
        mImageCount++;
        if (myImage.exists()) myImage.delete();
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(myImage);
            m.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

