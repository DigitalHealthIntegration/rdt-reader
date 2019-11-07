package com.iprd.rdtcamera;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.WARP_INVERSE_MAP;
import static org.opencv.imgproc.Imgproc.pyrDown;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.imgproc.Imgproc.warpPerspective;
import static org.opencv.video.Video.MOTION_EUCLIDEAN;
import static org.opencv.video.Video.MOTION_HOMOGRAPHY;
import static org.opencv.video.Video.findTransformECC;

public class ImageRegistration {

    public static Mat GetTransform(Mat refM, Mat insM) {
        Mat ref = new Mat();
        pyrDown(refM, ref);
        pyrDown(ref, ref);

        Mat ins = new Mat();
        pyrDown(insM, ins);
        pyrDown(ins, ins);

        Mat warpMatrix = getTransformation(ref, ins);

        return warpMatrix;
    }

    public static Mat getTransformation(Mat ref, Mat ins) {
        final int warp_mode = MOTION_EUCLIDEAN;
        Mat warpMatrix;
        if (warp_mode == MOTION_HOMOGRAPHY)
            warpMatrix = new Mat(3, 3, CV_32F);
        else
            warpMatrix = new Mat(2, 3, CV_32F);
        int numIter = 50;

        double terminationEps = 1e-3;
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, numIter, terminationEps);
        findTransformECC(ref, ins, warpMatrix, warp_mode, criteria, ins);
//       Mat ins_aligned = new Mat(ins.cols(), ins.rows(), CvType.CV_8U);
//        if (warp_mode != MOTION_HOMOGRAPHY)
//            // Use warpAffine for Translation, Euclidean and Affine
//            warpAffine(ins, ins_aligned, warpMatrix, ref.size(), INTER_LINEAR + WARP_INVERSE_MAP);
//        else
//            // Use warpPerspective for Homography
//            warpPerspective(ins, ins_aligned, warpMatrix, ref.size(),INTER_LINEAR + WARP_INVERSE_MAP);

        //Utils.SaveMatrix(ins_aligned,"warped");
        return warpMatrix;
    }
}
