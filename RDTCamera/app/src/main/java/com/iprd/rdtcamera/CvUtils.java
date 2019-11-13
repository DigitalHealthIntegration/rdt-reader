package com.iprd.rdtcamera;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.utils.Converters;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

public class CvUtils {

    private static Mat vector_Point2f_to_Mat(List<Point> pts) {
        int count=pts.size();
        Mat res = new Mat(1, count, CvType.CV_32FC2);
        float[] buff = new float[count * 2];
        for (int i = 0; i < count; i++) {
            Point p = pts.get(i);
            buff[i * 2] = (float) p.x;
            buff[i * 2 + 1] = (float) p.y;
        }
        res.put(0, 0, buff);
        return res;
    }

    public static Point warpPoint(Point point, Mat R){
        Point result= new Point();
        result.x = point.x * R.get(0,0)[0] + point.y *  R.get(0,1)[0]+  R.get(0,2)[0];
        result.y = point.x * R.get(1,0)[0] + point.y *  R.get(1,1)[0]+  R.get(1,2)[0];
        return result;
    }

    public static Mat scaleAffineMat(Mat warpmat, int level) {
        Mat warp= warpmat.clone();
        //level 4 mat
        int factor = 1<<level;
        //warp.put(0,1,warp.get(0,1)[0]*factor);
        //warp.put(1,0,warp.get(1,0)[0]*factor);
        warp.put(0,2,warp.get(0,2)[0]*factor);
        warp.put(1,2,warp.get(1,2)[0]*factor);
        return warp;
    }
    public static void PrintAffineMat(String s,Mat R){
        Log.i(s+"[0]",R.get(0,0)[0]+"x"+R.get(0,1)[0]+"x"+ R.get(0,2)[0]);
        Log.i(s+"[1]",R.get(1,0)[0]+"x"+R.get(1,1)[0]+"x"+ R.get(1,2)[0]);
    }

    public static double computePredictionLoss(List<Point> refPts, List<Point> insPts, Size imgSize) throws Exception
    {
        double loss=Double.MAX_VALUE;
        if(refPts.size()<3) throw new Exception("At least 3 points are required");
        if(refPts.size()!=insPts.size()) throw new Exception("Ref and insp should have same number of points");

        Mat src = vector_Point2f_to_Mat(refPts);
        Mat dst = vector_Point2f_to_Mat(insPts);
        Mat R= Video.estimateRigidTransform(src,dst,false);
        if(R.cols()<2 || R.rows()<1) return loss;

        List<Point> warpedPts= new ArrayList<>();

        for (Point p:insPts) {
            Point w= warpPoint(p,R);
            warpedPts.add(w);
            loss += Math.sqrt ((w.x-p.x)*(w.x-p.x) +(w.y-p.y)*(w.y-p.y));
        }
        return loss;
    }

    public static void preditionLossTest() throws Exception{
        List<Point> ref= new ArrayList<Point>();
        ref.add(new Point(100,0));
        ref.add(new Point(0,100));
        ref.add(new Point(-100,0));
        ref.add(new Point(0,-100));

        List<Point> ins= new ArrayList<Point>();
        ins.add(new Point(71,-71));
        ins.add(new Point(71,71));
        ins.add(new Point(-71,71));
        ins.add(new Point(71,-71));

        Size imgSize = new Size(1280,720);
        //test for identity
        double l=CvUtils.computePredictionLoss(ref,ins,imgSize);


    }

}
