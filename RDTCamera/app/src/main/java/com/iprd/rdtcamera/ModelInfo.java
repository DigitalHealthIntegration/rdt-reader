package com.iprd.rdtcamera;

public class ModelInfo {

    //public static final String mModelFileName="OD_360x640_10x19_slow.lite"; //1500 ms
    //public static final String mModelFileName="OD_360x640_10x19.lite"; //1000 ms
    //public static final String mModelFileName="OD_360x640_smaller.lite"; //1400 ms
    //public static final String mModelFileName="OD_360x640.lite"; //3500 ms
    //public static final String mModelFileName="OD_180x320_5x9.lite"; //280 ms
    //public static final String mModelFileName="OD_180x320.lite"; //480 ms
    //public static final String mModelFileName="OD_360x640_Scale_25.lite"; //1500 ms
    public static final String mModelFileName= "OD_180x320_newarch.lite";

    public static float deviationThresh=0.01f;

    public static int[] inputSize = {360,640};
    public static int[] aspectAnchors = new int[]{30, 70, 68, 68, 44, 74, 28, 52};
    public static int[] numberBlocks = new int[]{10,19};
    public static int pyrlevelcnt =1;

    static{
        if(mModelFileName.contains("180x320"))
        {
            inputSize = new int[]{180,320};
            aspectAnchors = new int[]{15, 35, 34,34, 22, 37, 14, 26};
            if(mModelFileName.contains("newarch")) {
                numberBlocks = new int[]{7, 16};
            }else{
                numberBlocks = new int[]{5, 9};
            }

            pyrlevelcnt =2;
        }
        else if(mModelFileName.contains("360x640"))
        {
            //this is already intitialized above
        }
        else {
            throw new RuntimeException("Unitialized model");
        }
    }



}
