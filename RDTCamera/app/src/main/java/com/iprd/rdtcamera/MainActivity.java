package com.iprd.rdtcamera;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
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
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.preference.PreferenceManager;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.core.Mat;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.Manifest.permission_group.CAMERA;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.floodFill;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "AndroidCameraApi";

    private ArrayList<AcceptanceStatus> mAcceptArray;
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private ImageView mRectView;

    private static final int SENSOR_ORIENTATION_DEFAULT_DEGREES = 90;
    private static final int SENSOR_ORIENTATION_INVERSE_DEGREES = 270;
    private static final SparseIntArray DEFAULT_ORIENTATIONS = new SparseIntArray();
    private static final SparseIntArray INVERSE_ORIENTATIONS = new SparseIntArray();
    public static Size CAMERA2_PREVIEW_SIZE = new Size(1280, 720);
    public static Size CAMERA2_IMAGE_SIZE = new Size(1280, 720);
    private Button preferenceSettingBtn;
    private TextView rdtDataToBeDisplay;
    private List<Surface> mSurfaces;

    static {
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_0, 90);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_90, 0);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_180, 270);
        DEFAULT_ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }
    static {
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_0, 270);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_90, 180);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_180, 90);
        INVERSE_ORIENTATIONS.append(Surface.ROTATION_270, 0);
    }
    private AutoFitTextureView mTextureView;
    private Size mPreviewSize;
    private Size mVideoSize;
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mPreviewSession;

    ImageReader mImageReader;
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
 //   private Semaphore mCameraOpenCloseLock = new Semaphore(1);
    private CaptureRequest.Builder mPreviewBuilder;

    private Integer mSensorOrientation;
    private double mTopTh,mBotTh;
    private short mShowImageData=0;
    public Config config = new Config();


    int idx;
    RdtAPI mRdtApi;
    byte[] mtfliteBytes = null;

    private boolean checkpermission(){
        System.out.println("..>>"+ WRITE_EXTERNAL_STORAGE);
        int res  = ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA);
        int res1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int res2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        return res1 == PackageManager.PERMISSION_GRANTED && res == PackageManager.PERMISSION_GRANTED && res2 == PackageManager.PERMISSION_GRANTED;
    }
    private void requestPermission(){
       /* if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE, Manifest.permission.CAMERA,ACCESS_FINE_LOCATION}, 200);
        }*/
        ActivityCompat.requestPermissions(this, new String[]{READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE,CAMERA, Manifest.permission.CAMERA}, 200);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if(!checkpermission()){
            requestPermission();
        }
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mTextureView = (AutoFitTextureView) findViewById(R.id.texture);

        mRectView = findViewById(R.id.rdtRect);
        rdtDataToBeDisplay = findViewById(R.id.rdtDataToBeDisplay);
        //rdtDataToBeDisplay.setTextColor(0x000000FF);
        mAcceptArray = new ArrayList<>();
        mAcceptArray.add(new AcceptanceStatus((short) 0,(short)0,(short)0,(short)0,(short)0,(short)0,(short)50,(short)50,(short)400,(short)50));
        mAcceptArray.add(new AcceptanceStatus((short) 0,(short)0,(short)0,(short)0,(short)0,(short)0,(short)100,(short)100,(short)500,(short)400));

        Config c = new Config();
        try {
            c.mTfliteB = ReadAssests();
        } catch (IOException e) {
            e.printStackTrace();
        }
        mRdtApi = new RdtAPI();
        mRdtApi.init(c);
        // preferences
        preferenceSettingBtn = (Button) findViewById(R.id.preferenceSettingBtn);
        preferenceSettingBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(MainActivity.this, MyPreferencesActivity.class);
                startActivity(i);
            }
        });
        ApplySettings();

        /// Set Torch button
        Switch sw = (Switch) findViewById(R.id.torch);
        sw.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                try{
                    if (isChecked) {
                        // The toggle is enabled
                        mPreviewBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);
                        mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, null);
                    } else {
                        // The toggle is disabled
                        mPreviewBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF);
                        mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, null);
                    }
                }catch (CameraAccessException e){
                    e.printStackTrace();
                }
            }
        });
        /// Set Torch button
        Switch saveData = (Switch) findViewById(R.id.saveData);
        saveData.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mRdtApi.setSaveImages(true);
                } else {
                    mRdtApi.setSaveImages(false);
                }
            }
        });

    }

    private void ApplySettings() {
        boolean mSaveNegativeData= false;
        try {
            SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(MainActivity.this);
            config.mMaxScale = Short.parseShort(prefs.getString("mMaxScale",  config.mMaxScale+""));
            config.mMinScale = Short.parseShort(prefs.getString("mMinScale", config.mMinScale+""));
            config.mXMin = Short.parseShort(prefs.getString("mXMin",  config.mXMin+""));
            config.mXMax = Short.parseShort(prefs.getString("mXMax", config.mXMax+""));
            config.mYMin = Short.parseShort(prefs.getString("mYMin", config.mYMin+""));
            config.mYMax = Short.parseShort(prefs.getString("mYMax", config.mYMax+""));
            config.mMinSharpness = Float.parseFloat(prefs.getString("mMinSharpness", config.mMinSharpness +""));
            config.mMaxBrightness = Float.parseFloat(prefs.getString("mMaxBrightness", config.mMaxBrightness+""));
            config.mMinBrightness = Float.parseFloat(prefs.getString("mMinBrightness", config.mMinBrightness+""));
            mTopTh = Float.parseFloat(prefs.getString("mTopTh", mTopTh+""));
            mBotTh = Float.parseFloat(prefs.getString("mBotTh", mBotTh+""));
            mShowImageData  = Short.parseShort(prefs.getString("mShowImageData", "0"));
            short t  = Short.parseShort(prefs.getString("mSaveNegativeData", mSaveNegativeData?"1":"0"));
            if(t!=0) mSaveNegativeData =true;
        }catch (NumberFormatException nfEx){//prefs.getString("mMinBrightness", "110.0f")
            Log.i("RDT","Exception in  Shared Pref switching to default");
            config.setDefaults();
            mTopTh = 0.9f;
            mBotTh = 0.7f;
            mShowImageData = 0;
            mSaveNegativeData = false;
        }
        mRdtApi.setConfig(config);
        mRdtApi.setTopThreshold(mTopTh);
        mRdtApi.setBottomThreshold(mBotTh);
        mRdtApi.mSaveNegativeData = mSaveNegativeData;
    }

    byte[] ReadAssests() throws IOException {
        InputStream is=getAssets().open("tflite.lite");
        mtfliteBytes=new byte[is.available()];
        is.read( mtfliteBytes);
        is.close();
        return mtfliteBytes;
    }

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private ImageReader.OnImageAvailableListener mImageAvailable = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
            //Log.d("Madhav","Comes in imagehandler");
            Image image = null;
            try {
                image = reader.acquireLatestImage();
                //final Mat captureMat = imageToRGBAMat(image);
                //Bitmap resultBitmap = Bitmap.createBitmap(captureMat.cols(), captureMat.rows(), Bitmap.Config.ARGB_8888);
                //Utils.matToBitmap(captureMat, resultBitmap);
                //Log.d("Image",captureMat.cols()+"x"+captureMat.rows());
                //Log.d("Image",image.getWidth()+"x"+image.getHeight());

            } catch(Exception e){

            } finally {
                if (image != null) {
                    image.close();
                }
            }
        }
    };

    private TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture,
                                              int width, int height) {
            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture,
                                                int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return true;
        }
        int count = 0;
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
            if(!mRdtApi.isInProgress()) {
                Bitmap capFrame = mTextureView.getBitmap();
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
            if(mShowImageData != 0){
                status.mSharpness = mRdtApi.mSharpness;
                status.mBrightness = mRdtApi.mBrightness;
            }
            long et = System.currentTimeMillis()-st;
            Log.i("Total Processing Time "," "+ et);
//            final AcceptanceStatus status = getCords(null);
            //final AcceptanceStatus status = getCords(mat);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    repositionRect(status);
                }
            });
            //Log.d("Madhav",capFrame.getWidth()+"x"+capFrame.getHeight() );
        }
    };

    private CameraCaptureSession.CaptureCallback mCaptureCallback
            = new CameraCaptureSession.CaptureCallback() {

        private void process(CaptureResult result) {
        }
    };

    private CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            startPreview();
            //////mCameraOpenCloseLock.release();
            if (null != mTextureView) {
                configureTransform(mTextureView.getWidth(), mTextureView.getHeight());
            }
        }
        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            //////mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
        }
        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            //////mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
        }
    };
    private static Size chooseVideoSize(Size[] choices) {
        for (Size size : choices) {
            Log.d("Madhav ",size.toString());
            if (size.getWidth() == size.getHeight() * 16 / 9 && size.getWidth() <= 1280) {
                return size;
            }
        }
        Log.e(TAG, "Couldn't find any suitable video size");
        return choices[choices.length - 1];
    }

    private boolean openCamera(int width, int height) {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        Log.e(TAG, "is camera open");
        try {
            String cameraId = manager.getCameraIdList()[0];
            // Add permission for camera and let user grant the permission
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return false;
            }

            CameraCharacteristics characteristics =
                    manager.getCameraCharacteristics(cameraId);

            StreamConfigurationMap map =
                    characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

            if (map == null) {
                Log.e(TAG, "No StreamConfigurationMap available");
                return false;
            }

            Size[] sizes = map.getOutputSizes(ImageReader.class);
            if (sizes.length == 0)
                return false;
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
            mVideoSize = closestPreviewSize;
            Size videoSize =closestPreviewSize;//= chooseOptimalSize(sizes, mVideoSize.getWidth(), mVideoSize.getHeight(),mVideoSize);
            mImageReader = ImageReader.newInstance(videoSize.getWidth(),
                    videoSize.getHeight(),
                    ImageFormat.YUV_420_888, 3);
            mImageReader.setOnImageAvailableListener(mImageAvailable,mBackgroundHandler);

            // Get all available size for the textureSurface preview window
            sizes = map.getOutputSizes(SurfaceTexture.class);
            // Get the optimal size for a preview window
            mPreviewSize = closestPreviewSize;//chooseOptimalSize(sizes, width, height,mVideoSize);

            // We fit the aspect ratio of TextureView to the size of preview we picked.
            int orientation = getResources().getConfiguration().orientation;
            if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
                mTextureView.setAspectRatio(
                        mPreviewSize.getWidth(), mPreviewSize.getHeight());
            } else {
                mTextureView.setAspectRatio(
                        mPreviewSize.getHeight(), mPreviewSize.getWidth());
            }
            configureTransform(width, height);
            manager.openCamera(cameraId, mStateCallback, null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Cannot access the camera.", e);
            return false;
        } catch (SecurityException e) {
            Log.e(TAG, "No access to camera device", e);
            return false;
        }
        return true;
    }

    /**
     * Configures the necessary {@link android.graphics.Matrix} transformation to `mTextureView`.
     * This method should not to be called until the camera preview size is determined in
     * openCamera, or until the size of `mTextureView` is fixed.
     *
     * @param viewWidth  The width of `mTextureView`
     * @param viewHeight The height of `mTextureView`
     */
        private void configureTransform(int viewWidth, int viewHeight) {

        if (null == mTextureView || null == mPreviewSize || null == this) {
            return;
        }
        int rotation = this.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }

    private void closeCamera() {
//        try {
            //////mCameraOpenCloseLock.acquire();
            closePreviewSession();
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            if (null != mImageReader) {
                mImageReader.close();
                mImageReader = null;
            }
//        } catch (InterruptedException e) {
//            throw new RuntimeException("Interrupted while trying to lock camera closing.");
//        } finally {
//            //////mCameraOpenCloseLock.release();
//        }
   }
    private void startPreview() {
        if (mCameraDevice == null ||!mTextureView.isAvailable() ||mPreviewSize == null || mImageReader == null) {
            return;
        }
        try {
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            texture.setDefaultBufferSize(mPreviewSize.getWidth(),
                    mPreviewSize.getHeight());
            mPreviewBuilder =
                    mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mSurfaces = new ArrayList<>();

            Surface previewSurface = new Surface(texture);
            mSurfaces.add(previewSurface);
            mPreviewBuilder.addTarget(previewSurface);

            Surface readerSurface = mImageReader.getSurface();
            mSurfaces.add(readerSurface);
            mPreviewBuilder.addTarget(readerSurface);

            mCameraDevice.createCaptureSession(mSurfaces,
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                            mPreviewSession = cameraCaptureSession;
                            updatePreview();
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession cameraCaptureSession){
                            Log.w(TAG, "Create capture session failed");
                        }
                    }, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void updatePreview() {
        if (mCameraDevice == null ||mPreviewBuilder == null || mBackgroundHandler == null|| mPreviewSession == null ) {
            return;
        }
        try {
            setUpCaptureRequestBuilder(mPreviewBuilder);
//            HandlerThread thread = new HandlerThread("CameraPreview");
//            thread.start();
            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void setUpCaptureRequestBuilder(CaptureRequest.Builder builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);

    }

    private AcceptanceStatus getCords(Mat mat){
        if(idx>=mAcceptArray.size()){
            idx=0;
        }
        return mAcceptArray.get(idx++);
    }

    AcceptanceStatus prevStat;
    long prevTime=0;
    private void repositionRect(AcceptanceStatus status){

        if(mRectView == null)return;
        if(status.mRDTFound){
            prevTime = System.currentTimeMillis();
            prevStat = status;
        }else {
            long curr = System.currentTimeMillis();
//            if ((curr - prevTime) > 5000) { //5 sec of timeout
//                Log.d("TimeDiff","curr " + curr + "prev " + prevTime +" = " +(curr - prevTime) );
                mRectView.setVisibility(View.INVISIBLE);
                if(rdtDataToBeDisplay != null) {
                    rdtDataToBeDisplay.setVisibility(View.INVISIBLE);
                }
                return;
//            }
        }

        mRectView.bringToFront();
        rdtDataToBeDisplay.bringToFront();
//        Log.d("ROI","Bounds "+status.mBoundingBoxX+"x"+status.mBoundingBoxY+" Position "+status.mBoundingBoxWidth+"x"+status.mBoundingBoxHeight);
        Point boundsRatio = new Point(prevStat.mBoundingBoxWidth*1.0/CAMERA2_PREVIEW_SIZE.getWidth(),prevStat.mBoundingBoxHeight*1.0/CAMERA2_PREVIEW_SIZE.getHeight()),
                positionRatio = new Point(prevStat.mBoundingBoxX*1.0/CAMERA2_PREVIEW_SIZE.getWidth(),prevStat.mBoundingBoxY*1.0/CAMERA2_PREVIEW_SIZE.getHeight());
//        Size boxBounds = new Size((int) Math.round(boundsRatio.x*mWindowSize.getWidth()),(int) Math.round(boundsRatio.y*mWindowSize.getHeight())),
//                boxPosition=new Size((int) Math.round(positionRatio.x*mWindowSize.getWidth()) , (int)Math.round(positionRatio.y*mWindowSize.getHeight()));
        ConstraintLayout.LayoutParams lp = (ConstraintLayout.LayoutParams) mRectView.getLayoutParams();
        /*lp.width = boxBounds.getWidth();
        lp.height=boxBounds.getHeight();
        lp.setMargins(boxPosition.getWidth(),boxPosition.getHeight(),0,0);*/
        lp.width = prevStat.mBoundingBoxWidth;
        lp.height=prevStat.mBoundingBoxHeight;
        lp.setMargins(prevStat.mBoundingBoxX,status.mBoundingBoxY,0,0);
        //Log.d("Box","Bounds "+lp.width+"x"+lp.height+" Position "+boxPosition.getWidth()+"x"+boxPosition.getHeight());
        mRectView.setLayoutParams(lp);
        mRectView.setVisibility(View.VISIBLE);
        if(mShowImageData !=0) {
            rdtDataToBeDisplay.setText("S[" + status.mSharpness+ "]\n"+"B[" + status.mBrightness+"]");
            rdtDataToBeDisplay.setVisibility(View.VISIBLE);
        }

    }

    @Override
    public void onResume() {
        super.onResume();
        mTextureView.setVisibility(View.VISIBLE);

        ApplySettings();
        //

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        startBackgroundThread();
        if (mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    /**
     * Starts a background thread and its {@link Handler}.
     */
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    /**
     * Stops the background thread and its {@link Handler}.
     */
    private void stopBackgroundThread() {
        if(mBackgroundHandler != null) {
            mBackgroundHandler.removeCallbacksAndMessages(null);
            mBackgroundThread.quitSafely();
            try {
                mBackgroundThread.join();
                mBackgroundThread = null;
                mBackgroundHandler = null;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }


    @Override
    public void onPause() {
        Log.d(TAG,"onPause called");
        mTextureView.setVisibility(View.GONE);
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        Log.d(TAG,"onDestroy called");
        closeCamera();
        stopBackgroundThread();
        super.onDestroy();
    }

    private void closePreviewSession() {
        if (mPreviewSession != null) {
            try {
                for(Surface surf:mSurfaces){
                    mPreviewBuilder.removeTarget(surf);
                }
                mPreviewSession.stopRepeating();
                mPreviewSession.abortCaptures();
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
            mPreviewSession.close();
            mPreviewSession = null;
        }
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults != null && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_DENIED && grantResults[1] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(MainActivity.this, "Sorry!!!, you can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
}