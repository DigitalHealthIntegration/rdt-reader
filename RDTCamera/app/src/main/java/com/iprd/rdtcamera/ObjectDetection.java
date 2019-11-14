package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class ObjectDetection {
    public static double mTopThreshold = 0.9;
    public static double mBottomThreshold = 0.7;
    Interpreter mTflite;
    Interpreter.Options tf_options = new Interpreter.Options();

    public long getTfliteTime() {
        return mTfLiteDuration;
    }

    public long getDivideTime() {
        return mDivideTime;
    }
    public long getROIFindingTime() {
        return mROIFindingTime;
    }
    long mROIFindingTime;
    long mTfLiteDuration;
    long mDivideTime;
    boolean mSaveImage=false;
    public void setSaveImages(boolean b){
        mSaveImage =b;
    }
    public boolean getSaveImages(){
        return mSaveImage;
    }


    public void setTopThreshold(double top){
        mTopThreshold = top;
    }
    public void setBottomThreshold(double bot){
        mBottomThreshold = bot;
    }
    ObjectDetection(MappedByteBuffer mappedbuffer){
       try {
            tf_options.setNumThreads(4);
            mTflite = new Interpreter(mappedbuffer, tf_options);
            if (mTflite != null) {
                Log.d("Loaded model File", "length = "+mappedbuffer.capacity());
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }
    ObjectDetection(byte[] bytes){
        //File modelFile = new File("/mnt/sdcard/mgd/tflite.lite");
        //mTflite = new Interpreter(modelFile);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bytes.length);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.put(bytes);
        try {
            tf_options.setNumThreads(4);
//            NnApiDelegate nnapiDel = new NnApiDelegate();
//            tf_options.addDelegate(nnapiDel);
            mTflite = new Interpreter(byteBuffer, tf_options);
            if (mTflite != null) {
                Log.d("Loaded model File", "length = "+ byteBuffer.capacity());
            }
        }catch(Exception e){
            e.printStackTrace();
        }
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
        int [] xy = new int[4];
        xy[0]= (int) (cx-w/2);
        xy[1]= (int) (cy-h/2);
        xy[2]= (int) (cx+w/2);
        xy[3]= (int) (cy+h/2);
        return xy;
    }

    public int[] nonMaxSupression(ArrayList<HashMap<Float, Vector<Integer>>> Top,ArrayList<HashMap<Float, Vector<Integer>>>  Bottom){
        int topn=Math.min(Top.size(),Bottom.size());
        //Log.d("detection","done");
        int [] roi = new int[4];
        int [] x1y1x2y2Top = {256,256,0,0};
        int [] x1y1x2y2Bot = {256,256,0,0};
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

    Rect update(Mat inputmat,Boolean [] rdt) {
        Rect ret = new Rect(-1, -1, -1, -1);
        try {
            long st  = System.currentTimeMillis();

            int[] roi = new int[4];
            int width = inputmat.cols();
            int height = inputmat.rows();
            //Resize image to 256x256 for the neural network
            Mat greyMat = new Mat();
            org.opencv.core.Size sz = new org.opencv.core.Size(256, 256);
            Imgproc.resize(inputmat, greyMat, sz);

            byte [] b = new byte[greyMat.channels()*greyMat.cols()*greyMat.rows()];
            greyMat.get(0,0,b); // get all the pixels

            float alpha = 1/255.0f;
            //Feed image pixels in normalized form to the input
            float[][][][] input = new float[1][256][256][1];
            int k = 0;
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    //double[] pixelvalue = res.get(i, j);
                    input[0][i][j][0] = ((b[k++]& 0xff)*alpha);// (pixelvalue[0]);// / 255.0);
                }
            }
            mDivideTime  = System.currentTimeMillis()-st;

            //Initialize output buffer
            float[][][][] output = new float[1][225][5][7];
            //Image to draw roi in
            Bitmap bmp = null;

            long startTime = System.currentTimeMillis();
            mTflite.run(input, output);
            mTfLiteDuration = System.currentTimeMillis()-startTime;

            mROIFindingTime = System.currentTimeMillis();
            //Log.i("mTfliteTime", String.valueOf(MethodeDuration));

//            AcceptanceStatus ret = update(mat.getNativeObjAddr());
//            if(null != ret ) Log.d("RDT FOUND ",ret.mRDTFound?"1":"0");
//            Log.d("TF","done");
            int[] anchors = new int[]{20, 10, 10, 20, 30, 30, 25, 20, 20, 25};
//           [[[20,10],[10,20],[30,30],[25,20],[20,25]]]

            ArrayList<HashMap<Float, Vector<Integer>>> vectorTableBottom = new ArrayList<HashMap<Float, Vector<Integer>>>();
            ArrayList<HashMap<Float, Vector<Integer>>> vectorTableTop = new ArrayList<HashMap<Float, Vector<Integer>>>();

            float resizeFactor = 256.0f / 15.0f;
            for (int row = 0; row < 15; row++) {
                for (int col = 0; col < 15; col++) {
                    for (int j = 0; j < 5; j++) {

                        float conf_0 = output[0][row * 15 + col][j][0];
                        float conf_1 = output[0][row * 15 + col][j][1];
                        if (conf_0 > mTopThreshold) {
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
                        } else if (conf_1 > mBottomThreshold) {
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
            if (vectorTableBottom.size() > 0 & vectorTableTop.size() > 0) {
                roi = nonMaxSupression(vectorTableTop, vectorTableBottom);
                int x1 = roi[0];
                int y1 = roi[1];
                int x2 = roi[2];
                int y2 = roi[3];
                //SaveROIImage(greyMat, x1, y1, x2, y2);
                ret.x = (int) (roi[0] / 256.0f * width);
                ret.y = (int) (roi[1] / 256.0f * height);
                ret.width = (int) (roi[2] / 256.0f * width);
                ret.height = (int) (roi[3] / 256.0f * height);
                rdt[0] = true;
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        finally {
            mROIFindingTime = System.currentTimeMillis()-mROIFindingTime;
        }
        if(mSaveImage) {
            if(rdt[0] == false){
               Utils.SaveROIImage(inputmat, ret.x,ret.y,ret.width,ret.height);
               Log.i("ROI",ret.x +"x" + ret.y + " " + ret.width+"x"+ret.height);
            }
        }
        return ret;
    }

}

