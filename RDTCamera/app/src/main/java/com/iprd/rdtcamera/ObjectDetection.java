package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.util.Log;
import android.widget.CompoundButton;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
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
import java.util.List;
import java.util.Map;
import java.util.Vector;

import static com.iprd.rdtcamera.ModelInfo.aspectAnchors;
import static com.iprd.rdtcamera.ModelInfo.inputSize;
import static com.iprd.rdtcamera.ModelInfo.numberBlocks;
import static com.iprd.rdtcamera.ModelInfo.pyrlevelcnt;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class ObjectDetection {
    private static float [] cannonicalArrow={121.0f,152.0f,182.0f};
    private static float [] cannonicalCpattern={596.0f,746.0f,895.0f};
    private static float [] cannonicalInfl={699.0f,874.0f,1048.0f};
    private static double ac_can = cannonicalCpattern[1]-cannonicalArrow[1];
    private static double ai_can = cannonicalInfl[1]-cannonicalArrow[1];


    private static Point  cannonicalA_C_Mid= new Point(449.0f,30.0f);
    private static Point  ref_A= new Point(cannonicalArrow[1]-cannonicalA_C_Mid.x,0.0f);
    private static Point  ref_C= new Point(cannonicalCpattern[1]-cannonicalA_C_Mid.x,0.0f);
    private static Point  ref_I= new Point(cannonicalInfl[1]-cannonicalA_C_Mid.x,0.0f);

    private static double ref_A_C = (cannonicalCpattern[1]-cannonicalArrow[1]);
    private static double scale = 0.0;
    private static double angleDegree = 0.0;

    public static double getMinError() {
        return minError;
    }

    private static double minError =100.0;
    private static double AvgConfbest =0.0;

    public void setSavePoints(boolean SavePoints) {
        this.mSavePoints = SavePoints;
    }
    boolean mSavePoints=false;

    private static int numberClasses = 31;
    private static int[] resizeFactor = {inputSize[0]/numberBlocks[0],inputSize[1]/numberBlocks[1]};
    private static float[] orientationAngles={0,22.5f,45,135,157.5f,180,202.5f,225,315,337.5f};
    protected ByteBuffer imgData =  ByteBuffer.allocateDirect(inputSize[0]*inputSize[1]*4);
    public Mat tmp_for_draw = null;
    Point C_arrow_predicted = new Point(0, 0);
    Point C_Cpattern_predicted = new Point(0, 0);
    Point C_Infl_predicted = new Point(0, 0);


    private static int numberAnchors=aspectAnchors.length/2;
    private double A_C_to_L = 1.624579124579125;
    private double L_to_W = 0.0601036269430052;
    private static double ref_hyp = 35 ;
    private boolean found =false;

    public static double mThreshold = 0.9;

    private float widthFactor = (float) (1.0/inputSize[1]*1280);
    private float heightFactor = (float) (1.0/inputSize[0]*720);
    Interpreter mTflite;
    Interpreter.Options tf_options = new Interpreter.Options();

    public void setTopThreshold(double top){
        mThreshold = top;
    }

    ObjectDetection(MappedByteBuffer mappedbuffer){
       try {
            tf_options.setNumThreads(4);
            mTflite = new Interpreter(mappedbuffer, tf_options);
            if (mTflite != null) {
                Log.d("Loaded model File", "length = ");
            }
           imgData.order(ByteOrder.nativeOrder());
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
                Log.d("Loaded model File", "length = ");
            }
            imgData.order(ByteOrder.nativeOrder());
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    class ScoreComparator implements Comparator<HashMap<Float,Vector<Float>>> {
        public int compare(HashMap<Float,Vector<Float>> s1,HashMap<Float,Vector<Float>> s2){

            Map.Entry<Float,Vector<Float>> entryS1 = s1.entrySet().iterator().next();
            Float keyS1 = entryS1.getKey();

            Map.Entry<Float,Vector<Float>> entryS2 = s2.entrySet().iterator().next();
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
    private void convertMattoTfLiteInput(Mat matInp)
    {
        imgData.rewind();
        for (int i = 0; i < inputSize[0]; ++i) {
            for (int j = 0; j < inputSize[1]; ++j) {
                imgData.putFloat((float) (matInp.get(i,j)[0]/255.0));
            }
        }
    }
    private int ArgMax(float[] inp)
    {
        float maxconf = 0.0f;
        int argmax=0;
        for (int i = 0; i < inp.length; ++i) {
            if (inp[i]>maxconf){
                argmax=i;
                maxconf=inp[i];
            }
        }
        return argmax;
    }
    //This input shooud be 1280x720 in following RDT direction and Grey scale
    //<<<----|| || || CCC Influenza
    Rect update(Mat inputmat,Boolean [] rdt) {
        found = false;
		if(mSavePoints) {
            tmp_for_draw = new Mat();
            Imgproc.cvtColor(inputmat, tmp_for_draw, Imgproc.COLOR_GRAY2RGBA, 4);
        }
        minError=100;
        Rect ret = new Rect(-1, -1, -1, -1);
        try {
            Mat greyMat = new Mat();
            if(pyrlevelcnt >=1)Imgproc.pyrDown(inputmat, greyMat);
            if(pyrlevelcnt >=2) Imgproc.pyrDown(greyMat, greyMat);

            //Feed image pixels in normalized form to the input
            float[][][][] input = new float[1][inputSize[0]][inputSize[1]][1];
            convertMattoTfLiteInput(greyMat);
//            for (int i = 0; i < inputSize[0]; i++) {
//                for (int j = 0; j < inputSize[1]; j++) {
//                    input[0][i][j][0] = (float) (greyMat.get(i,j)[0]/255.0);
//                }
//            }
            //Initialize output buffer
            float[][][][] output = new float[1][numberBlocks[0]*numberBlocks[1]][numberAnchors][numberClasses+4];
            //Image to draw roi in
            long startTime = System.currentTimeMillis();
            int[] dim = {1,inputSize[0],inputSize[1],1};
            mTflite.resizeInput(0,dim);
            mTflite.run(imgData, output);
            long MethodDuration = System.currentTimeMillis()-startTime;

            //Log.i("mTfliteTime", String.valueOf(MethodeDuration));
            ArrayList<HashMap<Float, Vector<Float>>> vectorTableArrow = new ArrayList<HashMap<Float, Vector<Float>>>();
            ArrayList<HashMap<Float, Vector<Float>>> vectorTableCpattern = new ArrayList<HashMap<Float, Vector<Float>>>();
            ArrayList<HashMap<Float, Vector<Float>>> vectorTableInfluenza = new ArrayList<HashMap<Float, Vector<Float>>>();
//            float resizeFactor = 256.0f / 15.0f;
            for (int row = 0; row < numberBlocks[0]; row++) {
                for (int col = 0; col < numberBlocks[1]; col++) {
                    for (int j = 0; j < numberAnchors; j++) {
                        int computedIndex = row * numberBlocks[1] + col;
                        int targetClass = ArgMax(Arrays.copyOfRange(output[0][computedIndex][j],0,31));
                        float confidence = output[0][computedIndex][j][targetClass];
                        if (confidence> mThreshold) {
                            int offsetStartIndex = numberClasses;
                            float cx = (float) ((col + 0.5) * resizeFactor[1] + output[0][computedIndex][j][offsetStartIndex] * inputSize[1])*widthFactor;
                            float cy = (float) ((row + 0.5) * resizeFactor[0] + output[0][computedIndex][j][offsetStartIndex+1] * inputSize[0])*heightFactor;
                            float w = (float) (aspectAnchors[j * 2+1]*Math.exp(output[0][computedIndex][j][offsetStartIndex+2] ))*widthFactor;
                            float h = (float) (aspectAnchors[j * 2 ]*Math.exp(output[0][computedIndex][j][offsetStartIndex+3]))*heightFactor;
                            Vector v = new Vector();
                            int typeOfFeat=targetClass/10;
                            float predictedOrientation =  orientationAngles[targetClass % 10];
                            v.add(cx);
                            v.add(cy);
                            v.add(w);
                            v.add(h);
                            v.add(predictedOrientation);
                            HashMap<Float, Vector<Float>> hMap = new HashMap<>();
                            hMap.put(confidence, v);
                            if (typeOfFeat==2){
                                vectorTableArrow.add(hMap);
                                if(mSavePoints) {
                                    Imgproc.circle(tmp_for_draw, new Point(cx,cy), 5, new Scalar(0, 0, 255), 2);
                                }
                            }
                            else if (typeOfFeat==1){
                                vectorTableCpattern.add(hMap);
                                if(mSavePoints) {
                                    Imgproc.circle(tmp_for_draw, new Point(cx,cy), 5, new Scalar(0, 255, 0), 2);
                                }
                            }
                            else if (typeOfFeat==0){
                                vectorTableInfluenza.add(hMap);
                                if(mSavePoints) {
                                    Imgproc.circle(tmp_for_draw, new Point(cx,cy), 5, new Scalar(255, 0, 0), 2);
                                }
                            }
                        }
                    }
                }
            }

            Log.d("size of answers","arrow:"+String.valueOf(vectorTableArrow.size())+"cpattern:"+String.valueOf(vectorTableCpattern.size())+"infl:"+String.valueOf(vectorTableInfluenza.size()));

            if (vectorTableArrow.size() > 0) {
                Collections.sort(vectorTableArrow, new ScoreComparator());
            }
            if (vectorTableCpattern.size() > 0) {
                Collections.sort(vectorTableCpattern, new ScoreComparator());
            }
            if (vectorTableInfluenza.size() > 0) {
                Collections.sort(vectorTableInfluenza, new ScoreComparator());
            }
            if (vectorTableArrow.size() > 0 & vectorTableCpattern.size() > 0&vectorTableInfluenza.size() > 0) {
                ret = locateRdt(vectorTableArrow, vectorTableCpattern,vectorTableInfluenza);

                rdt[0] = found;
            }
            if(found ){

                //grow width when it is too thin
                double grow_fact=0.25;
                if(ret.width*ret.height>0 && ret.height/ret.width <(1+grow_fact)*L_to_W)
                {
                    //remember height is the smaller direction
                    double adder=ret.height*grow_fact;
                    ret.height+=adder;
                    ret.y -=(adder/2);
                }

                if(ret.x <0) ret.x = 0;
                if(ret.y <0) ret.y = 0;
                if(ret.width <0) ret.width =0;
                if(ret.height <0) ret.height =0;

                if((ret.x+ret.width) >= inputmat.cols()) ret.width = inputmat.cols()-ret.x;
                if((ret.y+ret.height) >= inputmat.rows()) ret.height = inputmat.rows()-ret.y;

                if(mSavePoints) {
                    Imgproc.rectangle(tmp_for_draw, new Point(ret.x, ret.y), new Point(ret.x + ret.width, ret.y + ret.height), new Scalar(255, 0, 255), 1);
                }
            }
            if(mSavePoints) Utils.SavecentersImage(tmp_for_draw);

        } catch (Exception ex) {
            ex.printStackTrace();
        }
            if (rdt[0] == true) {
                Log.i("ROI", ret.x + "x" + ret.y + " " + ret.width + "x" + ret.height);
            }
        return ret;
    }
    private static double euclidianDistance(float[] p1,float[] p2){
        return Math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]));
    }

    private static Point rotatePoint(Point inp,double angleRadian, Point RDT_C){
        Point p = new Point(0,0);
        float s = (float) Math.sin(-angleRadian);
        float c = (float) Math.cos(-angleRadian);
        inp.x -= RDT_C.x;
        inp.y -= RDT_C.y;

        float xnew = (float) (inp.x * c - inp.y * s);
        float ynew = (float) (inp.x * s + inp.y * c);

        // translate point back:
        p.x = xnew + RDT_C.x;
        p.y = ynew + RDT_C.y;

        return p;
    }

    private static double lengthOfLine(Point p1, Point p2){
        return Math.sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
    }

    static double angleOfLine(Point p1, Point p2)
    {
        return Math.atan2((p2.y-p1.y),(p2.x-p1.x));
    }
    public static Point warpPoint(Point point, Mat R){
        Point result= new Point();
        result.x = point.x * R.get(0,0)[0] + point.y *  R.get(0,1)[0]+  R.get(0,2)[0];
        result.y = point.x * R.get(1,0)[0] + point.y *  R.get(1,1)[0]+  R.get(1,2)[0];
        return result;
    }

    public static Mat makeRMat(double scale, double theta, Point tr)
    {
        double cos_th=Math.cos(theta);
        double sin_th=Math.sin(theta);

        Mat R = new Mat(2,3, CvType.CV_32F);
        R.put(0,0,cos_th*scale);R.put(0,1,0-sin_th*scale);R.put(0,2,tr.x);
        R.put(1,0,sin_th*scale);R.put(1,1,cos_th*scale);R.put(1,2,tr.y);
        return R;
    }

    public static double detect2(float[] a, float[] c, float[] i, Point scale_rot)
    {
        Point3 orientations = new Point3(a[2],c[2],i[2]);
        return detect2(new Point(a[0],a[1]),new Point(c[0],c[1]),new Point(i[0],i[1]), orientations, scale_rot);
    }
    public static double detect2(Point a, Point c, Point i, Point3 orientations, Point out_scale_rot)
    {
        //rotation
        double th1=angleOfLine(a,c);
        double th2=angleOfLine(a,i);
        double theta=(th1+th2)/2;
        if(theta<0) theta+=2*Math.PI;

        //avoid feature orientations which are very different from theta
        double theta_deg=Math.toDegrees(theta);
        if(angle_constraint(orientations.x,theta_deg)||angle_constraint(orientations.y,theta_deg) ||angle_constraint(orientations.z,theta_deg))
        {
            return Double.MAX_VALUE;
        }

        //scale
        double ac=lengthOfLine(a,c);
        double ai=lengthOfLine(a,i);

        double s1=ac/ac_can;
        double s2=ai/ai_can;
        double scale=Math.sqrt(s1*s2);

        //avoid scales which are very different from each other
        double scale_disparity=s1/s2;
        if(scale_disparity>1.25 || scale_disparity<0.75)
        {
            return Double.MAX_VALUE;
        }

        //The inspection points rotate back so use -theta angle
        double cos_th=Math.cos(-1*theta);
        double sin_th=Math.sin(-1*theta);

        Mat R = new Mat(2,3, CvType.CV_32F);
        R.put(0,0,cos_th/scale);R.put(0,1,0-sin_th/scale);R.put(0,2,0);
        R.put(1,0,sin_th/scale);R.put(1,1,cos_th/scale);R.put(1,2,0);

        //Now warp the points
        Point a1=warpPoint(a,R);
        Point c1=warpPoint(c,R);
        Point i1=warpPoint(i,R);

        Point ac1_mid=new Point((a1.x+c1.x)/2,(a1.y+c1.y)/2);
        //translate back to 0,0
        a1=new Point(a1.x-ac1_mid.x,a1.y-ac1_mid.y);
        c1=new Point(c1.x-ac1_mid.x,c1.y-ac1_mid.y);
        i1=new Point(i1.x-ac1_mid.x,i1.y-ac1_mid.y);

        out_scale_rot.x=scale;
        out_scale_rot.y=theta;

        //compute the MSE
        return (lengthOfLine(ref_A,a1)+lengthOfLine(ref_C,c1)+lengthOfLine(ref_I,i1))/3;
    }

    private static boolean angle_constraint(double orientation, double theta_deg) {
        double T=30;

        double d=Math.abs(orientation-theta_deg);
        if(d>180) d=360-d;
        if(d>T) return true;
        return false;
    }

    public static double detect(float[] C_arrow, float[] C_Cpattern, float[] C_Infl){
        boolean found = false;
        /////


        //scale

        double A_C = euclidianDistance(C_arrow,C_Cpattern);
        double scale = ref_A_C/A_C;
        double y = C_Cpattern[1]-C_arrow[1];
        double x =C_Cpattern[0]-C_arrow[0];


        //rotate
        double angleRadian = Math.atan2(y,x) ;
        angleDegree = Math.toDegrees(angleRadian);

        if (angleDegree<0){
            angleDegree+=360;
        }

        //translate
        Point C_arrow_scaled = new Point(C_arrow[0]*scale,C_arrow[1]*scale);
        Point C_Cpattern_scaled = new Point(C_Cpattern[0]*scale,C_Cpattern[1]*scale);
        Point C_Infl_scaled = new Point(C_Infl[0]*scale,C_Infl[1]*scale);


        Point A_C_mid_pred = new Point(C_arrow_scaled.x+(C_Cpattern_scaled.x-C_arrow_scaled.x)/2,C_arrow_scaled.y+(C_Cpattern_scaled.y-C_arrow_scaled.y)/2);

        Point _diff = new Point(A_C_mid_pred.x-cannonicalA_C_Mid.x,A_C_mid_pred.y-cannonicalA_C_Mid.y);


        Point RDT_C= new Point();

        RDT_C.x = A_C_mid_pred.x;//+ref_hyp*Math.cos(angleRadian);
        RDT_C.y = A_C_mid_pred.y;//+ref_hyp*Math.sin(angleRadian);
//        Log.d("RDT_center","Center rdt_x "+RDT_C.x+" Center rdt_y "+RDT_C.y+" A_C pred x"+A_C_mid_pred.x+" A_C pred y"+A_C_mid_pred.y+" Angle "+angleDegree);



        Point C_arrow_rotated = rotatePoint(C_arrow_scaled,angleRadian,RDT_C);
        Point C_Cpattern_rotated = rotatePoint(C_Cpattern_scaled,angleRadian,RDT_C);
        Point C_Infl_rotated = rotatePoint(C_Infl_scaled,angleRadian,RDT_C);


        Point C_arrow_translated = new Point(C_arrow_rotated.x-_diff.x,C_arrow_rotated.y-_diff.y);
        Point C_Cpattern_translated = new Point(C_Cpattern_rotated.x-_diff.x,C_Cpattern_rotated.y-_diff.y);
        Point C_Infl_translated = new Point(C_Infl_rotated.x-_diff.x,C_Infl_rotated.y-_diff.y);

//        Log.d("C_arrow_","C_arrow_real_x "+cannonicalArrow[1]+" C_arrow_predicted_x "+C_arrow_translated.x+" Angle "+angleDegree+" scale "+scale);
//        Log.d("C_pattern_","C_pattern_real_x "+cannonicalCpattern[1]+" C_pattern_predicted_x "+C_Cpattern_translated.x);
//        Log.d("C_arrow_","C_Infl_real_x "+cannonicalInfl[1]+" C_Infl_predicted_x "+C_Infl_translated.x);

        double error_A = euclidianDistance(new float[]{cannonicalArrow[1],30.0f}, new float[]{(float) C_arrow_translated.x, (float) C_arrow_translated.y});
        double error_C = euclidianDistance(new float[]{cannonicalCpattern[1],30.0f}, new float[]{(float) C_Cpattern_translated.x, (float) C_Cpattern_translated.y});
        double error_I = euclidianDistance(new float[]{cannonicalInfl[1],30.0f}, new float[]{(float) C_Infl_translated.x, (float) C_Infl_translated.y});
//        Log.d("LME","Arrow "+error_A+" C "+error_C+" I "+error_I);
        double meanError = (error_A + error_C + error_I) / 3;

        return meanError;
    }



    public Rect locateRdt(ArrayList<HashMap<Float, Vector<Float>>> Arrow, ArrayList<HashMap<Float, Vector<Float>>>  Cpattern, ArrayList<HashMap<Float, Vector<Float>>>  Infl){
        //Log.d("detection","done");
        Rect roi = new Rect(-1, -1, -1, -1);
        boolean exit = false;
        found = false;
        Point3 minConf=new Point3(100.0,100.0,100.0);
        Point3 maxConf=new Point3(0,0,0);



        int cnt_arr = 0;
        int cnt_c=0;
        int cnt_i=0;
        float []C_arrow_best = new float[2];
        float []C_Cpattern_best=new float[2];
        float []C_infl_best=new float[2];
        Point best_scale_rot= new Point();
        Point scale_rot= new Point();
        while(cnt_arr<Arrow.size()){
            cnt_c=0;
            try{
                for (Map.Entry arrowElement : Arrow.get(cnt_arr).entrySet()) {
                    while(cnt_c<Cpattern.size()){
                        cnt_i=0;
                        float arrowconf = (float) arrowElement.getKey();
                        Vector cxcywha = (Vector) arrowElement.getValue();
                        float []C_arrow = {(float) cxcywha.get(0), (float) cxcywha.get(1), (float) cxcywha.get(4)};

                        for (Map.Entry cElement : Cpattern.get(cnt_c).entrySet()){

                            float Cconf = (float) cElement.getKey();
                            cxcywha = (Vector) cElement.getValue();
                            float []C_Cpattern = {(float) cxcywha.get(0), (float) cxcywha.get(1), (float) cxcywha.get(4)};

                            while(cnt_i<Infl.size()) {
                                for (Map.Entry iElement : Infl.get(cnt_i).entrySet()) {
                                    float infConf = (float) iElement.getKey();
                                    float AvgConf = (arrowconf+Cconf+infConf)/3;
                                    cxcywha = (Vector) iElement.getValue();
                                    float[] C_Inlf = {(float) cxcywha.get(0), (float) cxcywha.get(1), (float) cxcywha.get(4)};
                                    C_arrow_predicted.x = C_arrow[0];
                                    C_arrow_predicted.y = C_arrow[1];
                                    C_Cpattern_predicted.x = C_Cpattern[0];
                                    C_Cpattern_predicted.y = C_Cpattern[1];
                                    C_Infl_predicted.x = C_Inlf[0];
                                    C_Infl_predicted.y = C_Inlf[1];
                                    //Log.d("Cpattern", String.valueOf(C_Cpattern[0] + " inf index " + cnt_i + " INfl len " + Infl.size()));

                                    if(mSavePoints) {
                                        Imgproc.circle(tmp_for_draw, C_arrow_predicted, 5, new Scalar(0, 0, 255), 5);
                                        Imgproc.circle(tmp_for_draw, C_Cpattern_predicted, 5, new Scalar(0, 255, 0), 5);
                                        Imgproc.circle(tmp_for_draw, C_Infl_predicted, 5, new Scalar(255, 0, 0), 5);
                                    }
                                    double tmperror = detect2(C_arrow, C_Cpattern, C_Inlf,scale_rot);
                                    minConf.x = Math.min(minConf.x,arrowconf);
                                    minConf.y = Math.min(minConf.y,Cconf);
                                    minConf.z = Math.min(minConf.z,infConf);
                                    maxConf.x = Math.max(maxConf.x,arrowconf);
                                    maxConf.y = Math.max(maxConf.y,Cconf);
                                    maxConf.z = Math.max(maxConf.z,infConf);
                                    if (tmperror<minError) {
                                        if(Math.abs(minError-tmperror)>(Math.abs(AvgConf-AvgConfbest)+2)){
                                            minError=tmperror;
                                            found = true;
                                            //Log.d("Entered", "New min");
                                            C_arrow_best = C_arrow;
                                            C_Cpattern_best = C_Cpattern;
                                            C_infl_best = C_Inlf;
                                            best_scale_rot=scale_rot.clone();
                                            AvgConfbest=AvgConf;


                                        }


                                        //roi = new Rect((int)C_arrow[0],(int)C_arrow[1],50,50);
                                    }
                                }
                                cnt_i++;
                            }
                        }
                        cnt_c++;
                    }
                }
            }catch (IndexOutOfBoundsException e){
                Log.e("Error","Index out of bound exception");
                exit=true;
            }
            cnt_arr++;
        }
        List<MatOfPoint> matOfPoints = new ArrayList<>(3);
        matOfPoints.add(new MatOfPoint(new Point(C_arrow_best[0],C_arrow_best[1]),new Point(C_Cpattern_best[0],C_Cpattern_best[1]),new Point(C_infl_best[0],C_infl_best[1])));
        Imgproc.polylines(tmp_for_draw,matOfPoints,false,new Scalar(255,255,0,255),5);
        Log.i("COnfidence min max",minConf.toString()+"  "+maxConf.toString());
        //double scale=best_scale_rot.x;
        double angleRads=best_scale_rot.y;
        if(angleRads>Math.PI)
            angleRads-=Math.PI*2;
        double calculatedAngleRotation= Math.toDegrees(angleRads);

        Point rdt_c = new Point(C_arrow_best[0] + (C_Cpattern_best[0] - C_arrow_best[0]) / 2, C_arrow_best[1] + (C_Cpattern_best[1] - C_arrow_best[1]) / 2);

        Size sz = new Size();
        sz.width = ac_can * A_C_to_L *best_scale_rot.x;
        sz.height = sz.width* L_to_W;

        if(true) {
            rdt_c.x += ref_hyp * Math.cos(angleRads);
            rdt_c.y -= ref_hyp * Math.sin(angleRads);
        }

        RotatedRect rotatedRect = new RotatedRect(rdt_c, sz, calculatedAngleRotation);
        roi = rotatedRect.boundingRect();
        //Log.d("ROI:", "X : " + roi.x + "Y : " + roi.y + "W : " + roi.width + "H : " + roi.height);

        return roi;
    }
}

