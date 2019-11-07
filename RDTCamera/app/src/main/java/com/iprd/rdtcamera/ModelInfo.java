package com.iprd.rdtcamera;

public class ModelInfo {

    //public static final String mModelFileName="OD_180x320_5x9.lite";
//    public static final String mModelFileName="OD_180x320.lite";
//
//    //OD_180x320_5x9.lite
//    public static int[] inputSize = {180,320};
//    public static int[] aspectAnchors =new int[]{15, 35, 34,34, 11, 37, 14, 26};
//    public static int[] numberBlocks = new int[]{5,9};
//    public static float deviationThresh=0.1f;
//    public static int pyrlevelcnt =2;

    // public static final String mModelFileName="OD_360x640_10x19.lite";
    // public static final String mModelFileName="OD_360x640_10x19_slow.lite";
    public static final String mModelFileName="OD_360x640.lite";
    //public static final String mModelFileName="OD_360x640_smaller.lite";

    //OD_360x640_10x19 or _slow.lite
    public static int[] inputSize = {360,640};
    public static int[] aspectAnchors = new int[]{30, 70, 68, 68, 44, 74, 28, 52};
    public static int[] numberBlocks = new int[]{10,19};
    public static float deviationThresh=0.01f;
    public static int pyrlevelcnt =1;

}
