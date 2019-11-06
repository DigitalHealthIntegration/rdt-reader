package com.iprd.rdtcamera;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class ObjectDetection {
    private static float[] cannonicalArrow = {121.0f, 152.0f, 182.0f};
    private static float[] cannonicalCpattern = {596.0f, 746.0f, 895.0f};
    private static float[] cannonicalInfl = {699.0f, 874.0f, 1048.0f};
    private static float[] cannonicalA_C_Mid = {349.0f, 449.0f, 538.0f};
    private static double refRatio = (cannonicalInfl[1] - cannonicalArrow[1]) / (cannonicalCpattern[1] - cannonicalArrow[1]);
    Point C_arrow_predicted = new Point(0, 0);
    Point C_Cpattern_predicted = new Point(0, 0);
    Point C_Infl_predicted = new Point(0, 0);
    Mat tmp_for_draw = null;

    public void setSavePoints(boolean SavePoints) {
        this.mSavePoints = SavePoints;
    }

    boolean mSavePoints=false;

    //OD_180x320_5x9.lite
//    private static int[] inputSize = {180,320};
//    private static int[] aspectAnchors =new int[]{15, 35, 34,34, 11, 37, 14, 26};//Larger one is for 360x640 model  new int[]{30, 70, 68, 68, 44, 74, 28, 52};
//    private static int[] numberBlocks = new int[]{5,9};
//    private static float deviationThresh=0.1f;
//    private static int pyrlevelcnt =2;

    //    OD_360x640_10x19 or _slow.lite
    private static int[] inputSize = {360, 640};
    private static int[] aspectAnchors = new int[]{30, 70, 68, 68, 44, 74, 28, 52};
    private static int[] numberBlocks = new int[]{10, 19};
    private static float deviationThresh = 0.01f;
    private static int pyrlevelcnt = 1;

    private static int numberClasses = 31;
    private static int[] resizeFactor = {inputSize[0] / numberBlocks[0], inputSize[1] / numberBlocks[1]};
    private static float[] orientationAngles = {0, 22.5f, 45, 135, 157.5f, 180, 202.5f, 225, 315, 337.5f};
    protected ByteBuffer imgData = ByteBuffer.allocateDirect(inputSize[0] * inputSize[1] * 4);


    private static int numberAnchors = aspectAnchors.length / 2;
    private double calculatedAngleRotation = 0.0;
    private double A_C_to_L = 1.624579124579125;
    private double L_to_W = 0.0601036269430052;
    private double ref_hyp = 35;
    private boolean found = false;
    public double[] RDT_C = {0.0, 0.0};
    public static double mThreshold = 0.9;

    private float widthFactor = (float) (1.0 / inputSize[1] * 1280);
    private float heightFactor = (float) (1.0 / inputSize[0] * 720);
    Interpreter mTflite;
    Interpreter.Options tf_options = new Interpreter.Options();

    boolean mSaveImage = false;

    public void setSaveImages(boolean b) {
        mSaveImage = b;
    }

    public boolean getSaveImages() {
        return mSaveImage;
    }

    public org.opencv.core.RotatedRect rotatedRect;

    public void setTopThreshold(double top) {
        mThreshold = top;
    }

    ObjectDetection(MappedByteBuffer mappedbuffer) {
        try {
            tf_options.setNumThreads(4);
            mTflite = new Interpreter(mappedbuffer, tf_options);
            if (mTflite != null) {
                Log.d("Loaded model File", "length = ");
            }
            imgData.order(ByteOrder.nativeOrder());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    ObjectDetection(byte[] bytes) {
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
                Log.d("Loaded model File", "length = ");
            }
            imgData.order(ByteOrder.nativeOrder());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    class ScoreComparator implements Comparator<HashMap<Float, Vector<Float>>> {
        public int compare(HashMap<Float, Vector<Float>> s1, HashMap<Float, Vector<Float>> s2) {

            Map.Entry<Float, Vector<Float>> entryS1 = s1.entrySet().iterator().next();
            Float keyS1 = entryS1.getKey();

            Map.Entry<Float, Vector<Float>> entryS2 = s2.entrySet().iterator().next();
            Float keyS2 = entryS2.getKey();
            return Float.compare(keyS2, keyS1);
        }
    }

    int[] cxcy2xy(float cx, float cy, float w, float h) {
        int[] xy = new int[4];
        xy[0] = (int) (cx - w / 2);
        xy[1] = (int) (cy - h / 2);
        xy[2] = (int) (cx + w / 2);
        xy[3] = (int) (cy + h / 2);
        return xy;
    }

    private void convertMattoTfLiteInput(Mat matInp) {
        imgData.rewind();
        for (int i = 0; i < inputSize[0]; ++i) {
            for (int j = 0; j < inputSize[1]; ++j) {
                imgData.putFloat((float) (matInp.get(i, j)[0] / 255.0));
            }
        }
    }

    private int ArgMax(float[] inp) {
        float maxconf = 0.0f;
        int argmax = 0;
        for (int i = 0; i < inp.length; ++i) {
            if (inp[i] > maxconf) {
                argmax = i;
                maxconf = inp[i];
            }
        }
        return argmax;
    }

    //This input shooud be 1280x720 in following RDT direction and Grey scale
    //<<<----|| || || CCC Influenza
    Rect update(Mat inputmat, Boolean[] rdt) {
        if(mSavePoints) {
            tmp_for_draw = new Mat();
            Imgproc.cvtColor(inputmat, tmp_for_draw, Imgproc.COLOR_GRAY2RGBA, 4);
        }

        Rect ret = new Rect(-1, -1, -1, -1);
        try {
            Mat greyMat = new Mat();
            if (pyrlevelcnt >= 1) Imgproc.pyrDown(inputmat, greyMat);
            if (pyrlevelcnt >= 2) Imgproc.pyrDown(greyMat, greyMat);

            //Feed image pixels in normalized form to the input
            float[][][][] input = new float[1][inputSize[0]][inputSize[1]][1];
            convertMattoTfLiteInput(greyMat);
//            for (int i = 0; i < inputSize[0]; i++) {
//                for (int j = 0; j < inputSize[1]; j++) {
//                    input[0][i][j][0] = (float) (greyMat.get(i,j)[0]/255.0);
//                }
//            }
            //Initialize output buffer
            float[][][][] output = new float[1][numberBlocks[0] * numberBlocks[1]][numberAnchors][numberClasses + 4];
            //Image to draw roi in
            long startTime = System.currentTimeMillis();
            int[] dim = {1, inputSize[0], inputSize[1], 1};
            mTflite.resizeInput(0, dim);
            mTflite.run(imgData, output);
            long MethodDuration = System.currentTimeMillis() - startTime;

            //Log.i("mTfliteTime", String.valueOf(MethodeDuration));
            ArrayList<HashMap<Float, Vector<Float>>> vectorTableArrow = new ArrayList<HashMap<Float, Vector<Float>>>();
            ArrayList<HashMap<Float, Vector<Float>>> vectorTableCpattern = new ArrayList<HashMap<Float, Vector<Float>>>();
            ArrayList<HashMap<Float, Vector<Float>>> vectorTableInfluenza = new ArrayList<HashMap<Float, Vector<Float>>>();
//            float resizeFactor = 256.0f / 15.0f;
            for (int row = 0; row < numberBlocks[0]; row++) {
                for (int col = 0; col < numberBlocks[1]; col++) {
                    for (int j = 0; j < numberAnchors; j++) {
                        int computedIndex = row * numberBlocks[1] + col;
                        int targetClass = ArgMax(Arrays.copyOfRange(output[0][computedIndex][j], 0, 31));
                        float confidence = output[0][computedIndex][j][targetClass];
                        if (confidence > mThreshold) {
                            int offsetStartIndex = numberClasses;
                            float cx = (float) ((col + 0.5) * resizeFactor[1] + output[0][computedIndex][j][offsetStartIndex] * inputSize[1]) * widthFactor;
                            float cy = (float) ((row + 0.5) * resizeFactor[0] + output[0][computedIndex][j][offsetStartIndex + 1] * inputSize[0]) * heightFactor;
                            float w = (float) (aspectAnchors[j * 2 + 1] * Math.exp(output[0][computedIndex][j][offsetStartIndex + 2])) * widthFactor;
                            float h = (float) (aspectAnchors[j * 2] * Math.exp(output[0][computedIndex][j][offsetStartIndex + 3])) * heightFactor;

                            Vector v = new Vector();
                            int typeOfFeat = targetClass / 10;
                            float predictedOrientation = orientationAngles[targetClass % 10];
                            v.add(cx);
                            v.add(cy);
                            v.add(w);
                            v.add(h);
                            v.add(predictedOrientation);
                            HashMap<Float, Vector<Float>> hMap = new HashMap<>();
                            hMap.put(confidence, v);
                            if (typeOfFeat == 2) {
                                vectorTableArrow.add(hMap);
                            } else if (typeOfFeat == 1) {
                                vectorTableCpattern.add(hMap);
                            } else if (typeOfFeat == 0) {
                                vectorTableInfluenza.add(hMap);
                            }
                        }
                    }
                }
            }

            Log.d("size of answers", "arrow:" + String.valueOf(vectorTableArrow.size()) + "cpattern:" + String.valueOf(vectorTableCpattern.size()) + "infl:" + String.valueOf(vectorTableInfluenza.size()));

            if (vectorTableArrow.size() > 0) {
                Collections.sort(vectorTableArrow, new ScoreComparator());
            }
            if (vectorTableCpattern.size() > 0) {
                Collections.sort(vectorTableCpattern, new ScoreComparator());
            }
            if (vectorTableInfluenza.size() > 0) {
                Collections.sort(vectorTableInfluenza, new ScoreComparator());
            }
            if (vectorTableArrow.size() > 0 & vectorTableCpattern.size() > 0 & vectorTableInfluenza.size() > 0) {
                ret = locateRdt(vectorTableArrow, vectorTableCpattern, vectorTableInfluenza);
                rdt[0] = found;
            }
            if(mSavePoints)Utils.SavecentersImage(tmp_for_draw);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        if (true) {
            if (rdt[0] == true) {
                if(mSaveImage)Utils.SaveROIImage(inputmat, ret.x, ret.y, ret.x + ret.width, ret.y + ret.height);
                Log.i("ROI", ret.x + "x" + ret.y + " " + ret.width + "x" + ret.height);
            }
        }
        return ret;
    }

    private double euclidianDistance(float[] p1, float[] p2) {
        return Math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]));
    }

    private boolean detect(float[] C_arrow, float[] C_Cpattern, float[] C_Infl) {
        //C_arrow[0] = x C_arrow[1] = y
        //C_Cpattern[0] = x C_Cpattern[1] = y
        //C_Infl[0] = x C_Infl[1] = y

        boolean found = true;
        boolean borderCase = false;
        //scale
        double A_I = euclidianDistance(C_arrow, C_Infl);
        double A_C = euclidianDistance(C_arrow, C_Cpattern);
        double predictedRatio = A_I / A_C;
        double y = C_Cpattern[1] - C_arrow[1];
        double x = C_Cpattern[0] - C_arrow[0];
        //rotate
        double angleRadian = Math.atan2(y, x);
        double angleDegree = Math.toDegrees(angleRadian);
        if (angleDegree < 0) {
            angleDegree += 360;
        }
        if (C_arrow[2] == C_Infl[2] && C_arrow[2] == C_Cpattern[2] && C_Cpattern[2] == C_Infl[2]) {
            Log.d("Condition 1", "Passed!!!" + " Angle arrow : " + C_arrow[2] + "Angle computed : " + angleDegree);
            if (angleDegree > 350 && C_arrow[2] == 0) {
                C_arrow[2] = 360;
            }
            if (Math.abs(C_arrow[2] - angleDegree) < 22.5) {
                Log.d("Condition 2", "Passed!!!" + " Ref ratio : " + refRatio + "ratio computed : " + predictedRatio);
                if (Math.abs(predictedRatio - refRatio) < deviationThresh) {
                    Log.d("Condition 3", "Passed!!!");
                    found = true;
                    calculatedAngleRotation = angleDegree;
                }
            }
        }
        return found;
    }


    public Rect locateRdt(ArrayList<HashMap<Float, Vector<Float>>> Arrow, ArrayList<HashMap<Float, Vector<Float>>> Cpattern, ArrayList<HashMap<Float, Vector<Float>>> Infl) {
        //Log.d("detection","done");
        Rect roi = new Rect(-1, -1, -1, -1);
        boolean exit = false;
//        found = false;
        calculatedAngleRotation = 0.0;
        int cnt_arr = 0;
        int cnt_c = 0;
        int cnt_i = 0;

        float[] C_arrow_best;
        float[] C_Cpattern_best;
        float[] C_infl_best;
        while (cnt_arr<Arrow.size()) {
            cnt_c=0;
            try {
                for (Map.Entry arrowElement : Arrow.get(cnt_arr).entrySet()) {
                    float arrowconf = (float) arrowElement.getKey();
                    Vector cxcywha = (Vector) arrowElement.getValue();
                    float[] C_arrow = {(float) cxcywha.get(0), (float) cxcywha.get(1), (float) cxcywha.get(4)};


                    while (cnt_c < Cpattern.size()) {
                        cnt_i=0;
                        for (Map.Entry cElement : Cpattern.get(cnt_c).entrySet()) {
                            float Cconf = (float) cElement.getKey();
                            cxcywha = (Vector) cElement.getValue();
                            float[] C_Cpattern = {(float) cxcywha.get(0), (float) cxcywha.get(1), (float) cxcywha.get(4)};

                            while (cnt_i < Infl.size()) {
                                for (Map.Entry iElement : Infl.get(cnt_i).entrySet()) {
                                    float infConf = (float) iElement.getKey();
                                    cxcywha = (Vector) iElement.getValue();
                                    float[] C_Inlf = {(float) cxcywha.get(0), (float) cxcywha.get(1), (float) cxcywha.get(4)};
                                    C_arrow_predicted.x = C_arrow[0];
                                    C_arrow_predicted.y = C_arrow[1];
                                    C_Cpattern_predicted.x = C_Cpattern[0];
                                    C_Cpattern_predicted.y = C_Cpattern[1];
                                    C_Infl_predicted.x = C_Inlf[0];
                                    C_Infl_predicted.y = C_Inlf[1];

                                    Log.d("Cpattern", String.valueOf(C_Cpattern[0] + " inf index " + cnt_i + " INfl len " + Infl.size()));

                                    if(mSavePoints) {
                                        Imgproc.circle(tmp_for_draw, C_arrow_predicted, 5, new Scalar(0, 0, 255), 5);
                                        Imgproc.circle(tmp_for_draw, C_Cpattern_predicted, 5, new Scalar(0, 255, 0), 5);
                                        Imgproc.circle(tmp_for_draw, C_Infl_predicted, 5, new Scalar(255, 0, 0), 5);
                                    }
                                    found = detect(C_arrow, C_Cpattern, C_Inlf);

                                    if (found == true) {
                                        Log.d("Entered", "FOUND!!!");

                                        C_arrow_best = C_arrow;
                                        C_Cpattern_best = C_Cpattern;
                                        C_infl_best = C_Inlf;


                                        double[] A_C_mid_pred = {C_arrow_best[0] + (C_Cpattern_best[0] - C_arrow_best[0]) / 2, C_arrow_best[1] + (C_Cpattern_best[1] - C_arrow_best[1]) / 2};
                                        double A_C_pred = euclidianDistance(C_arrow_best, C_Cpattern_best);

                                        double L_predicted = A_C_pred * A_C_to_L;
                                        double W_predicted = L_predicted * L_to_W;
                                        double tmpangle = 0;
                                        if (calculatedAngleRotation > 180) {
                                            tmpangle = calculatedAngleRotation - 360;
                                        }
                                        double angleRads = Math.toRadians(tmpangle);
                                        RDT_C[0] = A_C_mid_pred[0] + ref_hyp * Math.cos(angleRads);
                                        RDT_C[1] = A_C_mid_pred[1] + ref_hyp * Math.sin(angleRads);
                                        Point rdt_c = new Point(RDT_C[0], RDT_C[1]);
                                        Size sz = new Size(L_predicted, W_predicted);
                                        rotatedRect = new org.opencv.core.RotatedRect(rdt_c, sz, calculatedAngleRotation);
                                        roi = rotatedRect.boundingRect();
                                        Log.d("ROI:", "X : " + roi.x + "Y : " + roi.y + "W : " + roi.width + "H : " + roi.height);
//                                roi = new Rect((int)C_arrow[0],(int)C_arrow[1],50,50);
//                                break;
                                    }
                                }
                                cnt_i += 1;
                            }

                        }
                        cnt_c+=1;
                    }
                }
                cnt_arr+=1;
            } catch (IndexOutOfBoundsException e) {
                exit = true;
            }
        }
        return roi;
    }
}

