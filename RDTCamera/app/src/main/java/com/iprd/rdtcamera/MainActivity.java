package com.iprd.rdtcamera;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.floodFill;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "AndroidCameraApi";
    private Activity self;
/*    private Button takePictureButton;
    private Button btn_on;
    private Button btn_off;
    private Boolean flashLightStatus = false;*/

    private AutoFitTextureView mtextureView;
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private Size CAMERA2_PREVIEW_SIZE = new Size(1280, 720);
    private Size CAMERA2_IMAGE_SIZE = new Size(1280, 720);
    private Size mWindowSize;

    private ArrayList<AcceptanceStatus> mAcceptArray;
    private String cameraId,mCameraId;
    private Boolean mSupportsTorchMode;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest captureRequest;
    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private ImageReader imageReader;
    private File file;
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private boolean mFlashSupported;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private ImageView mRectView;
    private CameraManager manager;
    long start_time;
    long end_time;
    int idx;
    rdtapi mRdtApi;
    private HandlerThread mOnImageAvailableThread;
    private Handler mOnImageAvailableHandler;


    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
//                    mOpenCvCameraView.enableView();
//                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private final ImageReader.OnImageAvailableListener mOnImageAvailableListener
            = new ImageReader.OnImageAvailableListener() {

        @Override
        public void onImageAvailable(ImageReader reader) {
            if (reader == null) {
                return;
            }

            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }
            //Log.d(TAG, "LOCAL FOCUS STATE: " + mFocusState + ", " + FocusState.FOCUSED);
        }

    };



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mRectView = findViewById(R.id.rdtRect);
        mAcceptArray = new ArrayList<>();
        mAcceptArray.add(new AcceptanceStatus((short) 0,(short)0,(short)0,(short)0,(short)0,(short)0,(short)50,(short)50,(short)400,(short)50));
        mAcceptArray.add(new AcceptanceStatus((short) 0,(short)0,(short)0,(short)0,(short)0,(short)0,(short)100,(short)100,(short)500,(short)400));
        assert mtextureView != null;
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int height = displayMetrics.heightPixels;
        int width = displayMetrics.widthPixels;
        self = this;
        mtextureView = (AutoFitTextureView) findViewById(R.id.texture);
        int orientation = getResources().getConfiguration().orientation;
//        Size mPreviewSize= new Size(1280,720);
//        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
//            mtextureView.setAspectRatio(mPreviewSize.getWidth(), mPreviewSize.getHeight());
//        } else {
//            mtextureView.setAspectRatio(mPreviewSize.getHeight(), mPreviewSize.getWidth());
//        }
        assert mtextureView != null;
        mWindowSize = new Size(displayMetrics.widthPixels,displayMetrics.heightPixels);
        int vidWidth = Math.round(((float)displayMetrics.heightPixels) * (9.0f/16.0f));

        mtextureView.setSurfaceTextureListener(textureListener);

        Config c = new Config();
        mRdtApi = new rdtapi();
        mRdtApi.init(c);

    }

private void resizeView(StreamConfigurationMap map) {
        if (map == null) {
            return;
        }
        Size closestPreviewSize = new Size(Integer.MAX_VALUE, (int) (Integer.MAX_VALUE * (9.0 / 16.0)));
        Size closestImageSize = new Size(Integer.MAX_VALUE, (int) (Integer.MAX_VALUE * (9.0 / 16.0)));
        for (Size size : Arrays.asList(map.getOutputSizes(ImageFormat.YUV_420_888))) {
            Log.d(TAG, "Available Sizes: " + size.toString());
            if (size.getWidth() * 9 == size.getHeight() * 16) { //Preview surface ratio is 16:9
                double currPreviewDiff = (CAMERA2_PREVIEW_SIZE.getHeight() * CAMERA2_PREVIEW_SIZE.getWidth()) - closestPreviewSize.getHeight() * closestPreviewSize.getWidth();
                double newPreviewDiff = (CAMERA2_PREVIEW_SIZE.getHeight() * CAMERA2_PREVIEW_SIZE.getWidth()) - size.getHeight() * size.getWidth();
                double currImageDiff = (CAMERA2_IMAGE_SIZE.getHeight()* CAMERA2_IMAGE_SIZE.getWidth()) - closestImageSize.getHeight() * closestImageSize.getWidth();
                double newImageDiff = (CAMERA2_IMAGE_SIZE.getHeight() * CAMERA2_IMAGE_SIZE.getWidth()) - size.getHeight() * size.getWidth();
                if (Math.abs(currPreviewDiff) > Math.abs(newPreviewDiff)) {
                    closestPreviewSize = size;
                }
                if (Math.abs(currImageDiff) > Math.abs(newImageDiff)) {
                    closestImageSize = size;
                }
            }
        }
        CAMERA2_IMAGE_SIZE = new Size(closestImageSize.getWidth(),closestImageSize.getHeight());
        CAMERA2_PREVIEW_SIZE= new Size(closestPreviewSize.getWidth(),closestPreviewSize.getHeight());

        int orientation = getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
            mtextureView.setAspectRatio(
                    closestPreviewSize.getWidth(), closestPreviewSize.getHeight());
        } else {
            mtextureView.setAspectRatio(
                    closestPreviewSize.getHeight(), closestPreviewSize.getWidth());
        }
        imageReader = ImageReader.newInstance(closestImageSize.getWidth(), closestImageSize.getHeight(),
                        ImageFormat.YUV_420_888, /*maxImages*/5);
        imageReader.setOnImageAvailableListener(
                mOnImageAvailableListener, mOnImageAvailableHandler);
    }

//    protected Vector reduceBBox(Vector bboxes,Vector scores,Vector indexBbox){
//        Vector filtered_bbox = new Vector();
//
//
//
//    }

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //open your camera here
            openCamera();
        }
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
            //configureTransform(width,height);
        }
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }
        long start_time = System.currentTimeMillis();
        int count =0;
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
            if (count++ % 10 == 0) {
    //            Bitmap frame = Bitmap.createBitmap(textureView.getWidth(), textureView.getHeight(), Bitmap.Config.ARGB_8888);
                Bitmap capFrame = mtextureView.getBitmap();
                Process(capFrame);
            }
        }

        void Process(final Bitmap capFrame) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    ProcessBitmap(capFrame);
                }
            }).start();
        }

        private void ProcessBitmap(Bitmap capFrame) {
            long st  = System.currentTimeMillis();
            final AcceptanceStatus status = mRdtApi.update(capFrame);
            long et = System.currentTimeMillis()-st;
            Log.d("Total Processing Time "," "+ et);
//            final AcceptanceStatus status = getCords(null);
            //final AcceptanceStatus status = getCords(mat);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    repositionRect(status);
                }
            });
        }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            //This is called when the camera is open
            Log.e(TAG, "onOpened");
            cameraDevice = camera;
            createCameraPreview();
        }
        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }
        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };
    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());

        mOnImageAvailableThread = new HandlerThread("OnImageAvailableBackgroud");
        mOnImageAvailableThread.start();
        mOnImageAvailableHandler = new Handler(mOnImageAvailableThread.getLooper());
    }
    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        mOnImageAvailableThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;

            mOnImageAvailableThread.join();
            mOnImageAvailableThread = null;
            mOnImageAvailableHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }


    protected void createCameraPreview() {
        try {
            SurfaceTexture texture = mtextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
//            captureRequestBuilder.addTarget(imageReader.getSurface());
            cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback(){
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    //The camera is already closed
                    if (null == cameraDevice) {
                        return;
                    }
                    // When the session is ready, we start displaying the preview.
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }
                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(MainActivity.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void setUpCameraOutputs(int width, int height) {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics
                        = manager.getCameraCharacteristics(cameraId);

                // We don't use a front facing camera in this sample.
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    continue;
                }

                // choose optimal size
                Size closestPreviewSize = new Size(Integer.MAX_VALUE, (int) (Integer.MAX_VALUE * (9.0 / 16.0)));
                Size closestImageSize = new Size(Integer.MAX_VALUE, (int) (Integer.MAX_VALUE * (9.0 / 16.0)));
                for (Size size : Arrays.asList(map.getOutputSizes(ImageFormat.YUV_420_888))) {
                    Log.d(TAG, "Available Sizes: " + size.toString());
                    if (size.getWidth() * 9 == size.getHeight() * 16) { //Preview surface ratio is 16:9
                        double currPreviewDiff = (CAMERA2_PREVIEW_SIZE.getHeight() * CAMERA2_PREVIEW_SIZE.getWidth()) - closestPreviewSize.getHeight() * closestPreviewSize.getWidth();
                        double newPreviewDiff = (CAMERA2_PREVIEW_SIZE.getHeight() * CAMERA2_PREVIEW_SIZE.getWidth()) - size.getHeight() * size.getWidth();
                        double currImageDiff = (CAMERA2_IMAGE_SIZE.getHeight() * CAMERA2_IMAGE_SIZE.getWidth()) - closestImageSize.getHeight() * closestImageSize.getWidth();
                        double newImageDiff = (CAMERA2_IMAGE_SIZE.getHeight() * CAMERA2_IMAGE_SIZE.getWidth()) - size.getHeight() * size.getWidth();
                        if (Math.abs(currPreviewDiff) > Math.abs(newPreviewDiff)) {
                            closestPreviewSize = size;
                        }
                        if (Math.abs(currImageDiff) > Math.abs(newImageDiff)) {
                            closestImageSize = size;
                        }
                    }
                }
                Log.d(TAG, "Selected sizes: " + closestPreviewSize.toString() + ", " + closestImageSize.toString());
                Size mPreviewSize = closestPreviewSize;
//                mImageReader = ImageReader.newInstance(closestImageSize.getWidth(), closestImageSize.getHeight(),
//                        ImageFormat.YUV_420_888, /*maxImages*/5);
//                mImageReader.setOnImageAvailableListener(
//                        mOnImageAvailableListener, mOnImageAvailableHandler);
                CAMERA2_IMAGE_SIZE = closestImageSize;
                CAMERA2_PREVIEW_SIZE = closestPreviewSize;
                // We fit the aspect ratio of TextureView to the size of preview we picked.
                int orientation = getResources().getConfiguration().orientation;
                if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                    // mtextureView.setAspectRatio(
                    //           mPreviewSize.getWidth(), mPreviewSize.getHeight());
                } else {
                    //    mtextureView.setAspectRatio(
                    //           mPreviewSize.getHeight(), mPreviewSize.getWidth());
                }
                mSupportsTorchMode = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                mCameraId = cameraId;
                return;
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            //toast("Unable to open the camera.");
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        Log.e(TAG, "is camera open");

        try {
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];

            // Add permission for camera and let user grant the permission
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Log.e(TAG, "openCamera X");
    }
    protected void updatePreview() {
        if(null == cameraDevice) {
            Log.e(TAG, "updatePreview error, return");
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (null != imageReader) {
            imageReader.close();
            imageReader = null;
        }
    }

    private void configureTransform(int viewWidth, int viewHeight) {
        Activity activity = self;
        if (null == mtextureView|| null == imageDimension || null == activity) {
            return;
        }
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, imageDimension.getHeight(), imageDimension.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / imageDimension.getHeight(),
                    (float) viewWidth / imageDimension.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }
        mtextureView.setTransform(matrix);
    }
    private AcceptanceStatus getCords(Mat mat){
        if(idx>=mAcceptArray.size()){
            idx=0;
        }
        return mAcceptArray.get(idx++);
    }


    private void repositionRect(AcceptanceStatus status){
        if(mRectView == null)return;
        if(!status.mRDTFound){
            mRectView.setVisibility(View.INVISIBLE);
            return;
        }
        mRectView.bringToFront();
        Log.d("ROI","Bounds "+status.mBoundingBoxX+"x"+status.mBoundingBoxY+" Position "+status.mBoundingBoxWidth+"x"+status.mBoundingBoxHeight);

        Point boundsRatio = new Point(status.mBoundingBoxWidth*1.0/CAMERA2_PREVIEW_SIZE.getWidth(),status.mBoundingBoxHeight*1.0/CAMERA2_PREVIEW_SIZE.getHeight()),
                positionRatio = new Point(status.mBoundingBoxX*1.0/CAMERA2_PREVIEW_SIZE.getWidth(),status.mBoundingBoxY*1.0/CAMERA2_PREVIEW_SIZE.getHeight());
        Size boxBounds = new Size((int) Math.round(boundsRatio.x*mWindowSize.getWidth()),(int) Math.round(boundsRatio.y*mWindowSize.getHeight())),
                boxPosition=new Size((int) Math.round(positionRatio.x*mWindowSize.getWidth()) , (int)Math.round(positionRatio.y*mWindowSize.getHeight()));
        ConstraintLayout.LayoutParams lp = (ConstraintLayout.LayoutParams) mRectView.getLayoutParams();
        /*lp.width = boxBounds.getWidth();
        lp.height=boxBounds.getHeight();
        lp.setMargins(boxPosition.getWidth(),boxPosition.getHeight(),0,0);*/
        lp.width = status.mBoundingBoxWidth;
        lp.height=status.mBoundingBoxHeight;
        lp.setMargins(status.mBoundingBoxX,status.mBoundingBoxY,0,0);
        Log.d("Box","Bounds "+lp.width+"x"+lp.height+" Position "+boxPosition.getWidth()+"x"+boxPosition.getHeight());
        mRectView.setLayoutParams(lp);
        mRectView.setVisibility(View.VISIBLE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(MainActivity.this, "Sorry!!!, you can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    @Override
    protected void onResume() {
        super.onResume();
        Log.e(TAG, "onResume");

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        startBackgroundThread();
        if (mtextureView.isAvailable()) {
            openCamera();
        } else {
            mtextureView.setSurfaceTextureListener(textureListener);
        }
    }
    @Override
    protected void onPause() {
        Log.e(TAG, "onPause");
        //closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    //    public native void init(com.iprd.rdtcamera.Config c );
//    public native com.iprd.rdtcamera.AcceptanceStatus update(long m);
//    public native void setConfig(com.iprd.rdtcamera.Config c);
 //   static {
 //       System.loadLibrary("opencv_java");
//    }
}