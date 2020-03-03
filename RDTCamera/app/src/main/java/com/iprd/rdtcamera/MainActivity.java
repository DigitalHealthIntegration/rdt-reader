package com.iprd.rdtcamera;


import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.preference.CheckBoxPreference;
import android.preference.PreferenceManager;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.logging.Logger;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.Manifest.permission_group.CAMERA;
import static com.iprd.rdtcamera.AcceptanceStatus.GOOD;
import static com.iprd.rdtcamera.AcceptanceStatus.TOO_HIGH;
import static com.iprd.rdtcamera.Httpok.mHttpURL;
import static com.iprd.rdtcamera.ModelInfo.mModelFileName;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.floodFill;
import android.os.Vibrator;


public class MainActivity extends AppCompatActivity implements AsyncResponse{
    private boolean goodImageFlag;
    private static final String TAG = "AndroidCameraApi";
    public static final String MY_PREFS_NAME = "MyPrefsFile";
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private ImageView mRectView;
    private ImageView mRdtView,mTrackedView;
    private ImageView disRdtResultImage,mWarpedImage;
    Button mGetResult;
    Button startBtn;
    TextView mResultView,mMotionText,mStatusView;
    Boolean isPreviewOff = false;
    Boolean shouldOffTorch = false;
    Boolean isFlashRequired = false;
    SurfaceView surfaceView;
    private Paint paint;
    public static Paint p = new Paint();
    private Paint textPaint = new Paint();
    private Vibrator Vibobj;
    Boolean isfired = false;

    private Paint transparentPaint;
    long timeSinceLastChecked= System.currentTimeMillis();

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    SharedPreferences prefs;
    ProgressBar mCyclicProgressBar;

    private static final int SENSOR_ORIENTATION_DEFAULT_DEGREES = 90;
    private static final int SENSOR_ORIENTATION_INVERSE_DEGREES = 270;
    private static final SparseIntArray DEFAULT_ORIENTATIONS = new SparseIntArray();
    private static final SparseIntArray INVERSE_ORIENTATIONS = new SparseIntArray();
    public static Size CAMERA2_PREVIEW_SIZE = new Size(1080,1920);//1280, 720);
    public static Size CAMERA2_IMAGE_SIZE = new Size(1080,1920);//1280, 720);
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
    private SurfaceHolder mHolder;
    private ImageReader mImageReader;
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    private int left,top,right,bottom;
    private CaptureRequest.Builder mPreviewBuilder;

    private Integer mSensorOrientation;
    AcceptanceStatus status;
    private short mShowImageData = 0;
    public Switch mode;
    public Switch torch;
    public Switch saveData;
    boolean isGridDispaly;
    TableLayout gridTable;
    byte[] mImageBytes = null;

    int idx;
    RdtAPI mRdtApi = null;
    private RdtAPI.RdtAPIBuilder rdtAPIBuilder = null;

    private boolean checkpermission() {
        int res = ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA);
        int res1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int res2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        return res1 == PackageManager.PERMISSION_GRANTED && res == PackageManager.PERMISSION_GRANTED && res2 == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE,CAMERA, Manifest.permission.CAMERA}, 200);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (!checkpermission()) {
            requestPermission();
        }

        paint = new Paint();
        paint.setColor(0xcc000000);
        transparentPaint = new Paint();
        transparentPaint.setColor(getResources().getColor(android.R.color.transparent));
        transparentPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40);
        super.onCreate(savedInstanceState);
        mFile = new File(this.getExternalFilesDir(null), "pic.jpg");
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        mTextureView = (AutoFitTextureView) findViewById(R.id.texture);
        surfaceView = (SurfaceView) findViewById(R.id.surfaceView);
        surfaceView.setZOrderOnTop(true);
        mHolder = surfaceView.getHolder();
        mHolder.setFormat(PixelFormat.TRANSPARENT);
        mGetResult = findViewById(R.id.getResult);
        Context c = getApplicationContext();
        prefs = this.getSharedPreferences("MyPrefsFile", MODE_PRIVATE);
        gridTable = findViewById(R.id.gridTable);
        gridTable.setVisibility(View.VISIBLE);

        mRdtView = findViewById(R.id.RdtDetectImage);
        mTrackedView = findViewById(R.id.RdtTrackedImage);
        mRectView = findViewById(R.id.rdtRect);
        mMotionText = findViewById(R.id.MotionText);
        mWarpedImage = findViewById(R.id.RdtWarpImage);
        mStatusView = findViewById(R.id.Status);

        disRdtResultImage = findViewById(R.id.disRdtResultImage);
        rdtDataToBeDisplay = findViewById(R.id.rdtDataToBeDisplay);
        mCyclicProgressBar = findViewById(R.id.loader);

        mResultView = findViewById(R.id.ResultView);
        mResultView.bringToFront();

        disRdtResultImage.getLayoutParams().height = 500;
        disRdtResultImage.getLayoutParams().width = 100;
        disRdtResultImage.requestLayout();

        Vibobj = (Vibrator) this.getSystemService(VIBRATOR_SERVICE);
        startBtn = findViewById(R.id.startBtn);
        preferenceSettingBtn = (Button) findViewById(R.id.preferenceSettingBtn);
        preferenceSettingBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(MainActivity.this, MyPreferencesActivity.class);
                startActivity(i);
            }
        });

        byte[] mTfliteB = null;
        MappedByteBuffer mMappedByteBuffer = null;
        try {
            mTfliteB = ReadAssests();
            mMappedByteBuffer = Utils.loadModelFile(getAssets(),mModelFileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        rdtAPIBuilder = new RdtAPI.RdtAPIBuilder();
        rdtAPIBuilder = rdtAPIBuilder.setModel(mMappedByteBuffer);
        mShowImageData = Utils.ApplySettings(this,rdtAPIBuilder,null);
        mRdtApi = rdtAPIBuilder.build();
        mRdtApi.setmShowPip(true);

        //call the setter for saving functions
        Utils.ApplySettings(this,null,mRdtApi);

        // Set Torch button
        torch = (Switch) findViewById(R.id.torch);
        torch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                isFlashRequired = isChecked;
                System.out.println(">>>>>>>>>>>"+isfired);

                try{
                    if (shouldOffTorch == false && mPreviewSession != null ){
                        if (isChecked) {
                            // The toggle is enabled
                                mPreviewBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);
                                mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, mBackgroundHandler);

                        } else {
                            // The toggle is disabled
                            mPreviewBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF);
                            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, mBackgroundHandler);
                        }
                    } else {
                        shouldOffTorch = false;
                        if(torch.isChecked())
                            torch.setChecked(false);
                            // torch.setEnabled(false);
                        isFlashRequired = false;
                        }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

        mGetResult.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                mGetResult.setEnabled(false);
                mResultView.setText("");
                mResultView.setTextColor(Color.BLACK);
                mResultView.setVisibility(View.INVISIBLE);
                mImageBytes = null;
                handlerCall = true;
                mResultView.setText("");
                getRDTResultData();
                startBtn.setEnabled(true);
                mResultView.setTextColor(Color.BLACK);
                mResultView.setVisibility(View.INVISIBLE);
                mGetResult.setVisibility(View.INVISIBLE);
                shouldOffTorch = true;

                if(torch.isChecked())
                    torch.setChecked(false);
            }
        });
        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                timeSinceLastChecked=0;
                startPreview();
                startBtn.setEnabled(false);
                progressbar(false);
                shouldOffTorch = false;
                torch.setEnabled(true);
            }
        });
    }
    public void Continue() {

        startBtn.performClick();
        mWarpedImage.setImageBitmap(null);
        mWarpedImage.setVisibility(View.INVISIBLE);
    }
    public void rdtFound(boolean found,String msg){
        Canvas canvas = mHolder.lockCanvas();
        if (canvas == null) {
            Log.e(TAG, "Cannot - onto the canvas as it's null");
        } else {
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
            canvas.drawColor(Color.TRANSPARENT);
            canvas.drawRect(0, 0, mTextureView.getRight(), mTextureView.getBottom(), paint);
            if(!isfired) {
                if (found) {
                    p.setColor(Color.rgb(0, 200, 50));
                } else {
                    p.setColor(Color.rgb(255,25,25));
                }
            }
            p.setStrokeWidth(20);
            p.setStyle(Paint.Style.STROKE);

            canvas.drawRect(left, top, right, bottom, p);
            canvas.drawRect(left, top, right, bottom, transparentPaint);
            String[] msgString = msg.split("\\.\\.");
            float y = top*0.25f;
            for(String str : msgString){
                canvas.drawText(str.trim().toUpperCase(), left, y, textPaint);
                y += textPaint.descent() - textPaint.ascent();
            }
            if (mImageBytes!=null && mResultView != null ){
                if(disRdtResultImage.getDrawable() != null)
                    canvas.drawRect(disRdtResultImage.getLeft(), disRdtResultImage.getTop(), disRdtResultImage.getRight(), disRdtResultImage.getBottom(), transparentPaint);
                    canvas.drawRect(mResultView.getLeft(), mResultView.getTop(), mResultView.getRight(), mResultView.getBottom(), transparentPaint);

            }
            mHolder.unlockCanvasAndPost(canvas);
        }
    }

    void progressbar(boolean isVisible) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mCyclicProgressBar.setVisibility(isVisible ? View.VISIBLE : View.INVISIBLE);
                if(isVisible){
                    if(torch.isChecked()){
                        torch.setChecked(false);
                    }
                    torch.setEnabled(false);
                }
            }
        });
    }

    byte[] ReadAssests() throws IOException {
        byte[] mtfliteBytes = null;
        InputStream is = getAssets().open(mModelFileName);
        mtfliteBytes = new byte[is.available()];
        is.read(mtfliteBytes);
        is.close();
        return mtfliteBytes;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    private File mFile;
    private boolean handlerCall = false;
    public int i =0 ;

    private ImageReader.OnImageAvailableListener mImageAvailable = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
                Image image = null;
            try {
                image = reader.acquireLatestImage();
            } catch (Exception e) {
                e.printStackTrace();

            } finally {
                if (image != null) {
                    image.close();
                }
            }
        }
    };

    private TextureView.SurfaceTextureListener mSurfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture,int width, int height) {
            left = (int) (0.25*mTextureView.getWidth());
            top = (int) (0.1*mTextureView.getHeight());
            bottom = mTextureView.getHeight() - (int) (0.1*mTextureView.getHeight());
            right = mTextureView.getWidth() - (int) (0.25*mTextureView.getWidth());

            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return true;
        }

        private Semaphore sem = new Semaphore(1);

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
            torch.setVisibility(View.VISIBLE);

            if (!mRdtApi.isInprogress() && sem.tryAcquire(1) == true) {
                try {
                    Bitmap capFrame = mTextureView.getBitmap();
                    Process(capFrame);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        void Process(final Bitmap capFrame) {
            ProcessBitmap(capFrame);
        }

        private void ProcessBitmap(Bitmap capFrame) {
            if(!isfired) {
                status = mRdtApi.checkFrame(capFrame);
            }
            if (mShowImageData != 0) {
                status.mSharpness = mRdtApi.getSharpness();
                status.mBrightness = mRdtApi.getBrightness();
            }
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    String textTodisp = "RDT not found";
                    if (status.mRDTFound && !isfired) {
                        textTodisp = "RDT found .. ";
                        if (status.mInfo.mScale>0.6){
                            textTodisp  += "scale is good ..";
                        }else if (status.mInfo.mScale<0.6){
                            textTodisp  += "slowly bring camera closer ..";
                            timeSinceLastChecked = 0;
                        }
                        if (status.mInfo.mBrightness>120 && status.mInfo.mBrightness<200){
                            textTodisp  += "brightness is good.. hold steady";
                        }else if (status.mInfo.mBrightness<120) {
                            timeSinceLastChecked = 0;
                            textTodisp  += "brightness is low ..";
                        }else if (status.mInfo.mBrightness>200) {
                            timeSinceLastChecked = 0;
                            textTodisp  += "brightness is high ..";
                        }else{timeSinceLastChecked =0L;}
                        Log.i("TIME since last check", String.valueOf(timeSinceLastChecked));
                        Log.i("Scale", String.valueOf(status.mInfo.mScale));
                        if (status.mInfo.mScale>0.6 && status.mInfo.mBrightness>120 && status.mInfo.mBrightness<200 ){
                            rdtFound(true,textTodisp);
                            if (timeSinceLastChecked==0){
                                timeSinceLastChecked = System.currentTimeMillis();
                            }else if (System.currentTimeMillis()-timeSinceLastChecked > 2000) {
                                status = new AcceptanceStatus();
                                mRdtApi.mStatus =new AcceptanceStatus();
                                mRdtApi.mStatus2 =new AcceptanceStatus();

                                timeSinceLastChecked = 0L;
                                isfired = true;
                                mWarpedImage.setVisibility(View.VISIBLE);

                                p.setColor(Color.rgb(104,104,104));
                                mGetResult.performClick();
                            }
                        }
                    } else{
                        if(isfired)
                            textTodisp = "";

                        rdtFound(false,textTodisp);
                        timeSinceLastChecked =0L;
                    }
                }
            });
            sem.release();
        }
    };

    private CameraCaptureSession.CaptureCallback mCaptureCallback = new CameraCaptureSession.CaptureCallback() {

        private void process(CaptureResult result) {
        }
    };

    private CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            startPreview();
            if (null != mTextureView) {
                configureTransform(mTextureView.getWidth(), mTextureView.getHeight());
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            mCameraDevice = null;
        }
    };

    private static Size chooseVideoSize(Size[] choices) {
        for (Size size : choices) {
            Log.d("Camera ", size.toString());
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
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return false;
            }

            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

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
            for (Size size : Arrays.asList(map.getOutputSizes(ImageFormat.YUV_420_888))) {//YUV_420_888//YUV_420_888
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
            Size videoSize = closestPreviewSize;
            mImageReader = ImageReader.newInstance(videoSize.getWidth(), videoSize.getHeight(), ImageFormat.YUV_420_888, 5);

            mImageReader.setOnImageAvailableListener(mImageAvailable, mBackgroundHandler);

            mPreviewSize = closestPreviewSize;
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

    public void rdtResults(byte[] bytes) {
        OutputStream output = null;
        try {
            String urlString = prefs.getString("rdtCheckUrl", mHttpURL);
            String guid = String.valueOf(java.util.UUID.randomUUID());
            String metaDataStr = "{\"UUID\":" + "\"" + guid + "\",\"Quality_parameters\":{\"brightness\":\"10\"},\"RDT_Type\":\"Flu_Audere\",\"Include_Proof\":\"True\"}";
            try {
                Httpok mr = new Httpok("img.jpg", bytes, urlString, metaDataStr, mCyclicProgressBar, disRdtResultImage, mResultView);
                mr.delegate = this;
                mr.setCtx(getApplicationContext());
                mr.execute();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } finally {
            if (null != output) {
                try {
                    Continue();
                    output.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    public static Bitmap RotateBitmap(Bitmap source, float angle)
    {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }
    private ImageReader reader1 = null;
    private CaptureRequest.Builder captureBuilder = null;
    private CameraCaptureSession mSession;
    private Boolean isInnerFlashOn = false;
    private void getRDTResultData(){
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try{
           // handlerCall = true;
            String cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            Size[] jpegSizes = null;
            if (characteristics != null) {
                jpegSizes = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP).getOutputSizes(ImageFormat.JPEG);
            }

            int width = 600;
            int height = 400;
            if (jpegSizes != null && 0 < jpegSizes.length) {
                width = jpegSizes[0].getWidth()-100;
                height = jpegSizes[0].getHeight()-100;
            }

            reader1 = ImageReader.newInstance(width, height, ImageFormat.JPEG, 1);
            List<Surface> outputSurfaces = new ArrayList<Surface>(2);
            outputSurfaces.add(reader1.getSurface());
            outputSurfaces.add(new Surface(mTextureView.getSurfaceTexture()));
            captureBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);//TEMPLATE_PREVIEW
            captureBuilder.set(CaptureRequest.JPEG_QUALITY, (byte) 90);
            captureBuilder.addTarget(reader1.getSurface());
            int rotation = getWindowManager().getDefaultDisplay().getRotation();

            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(rotation));
            captureBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);//

            if(isFlashRequired) {
                isInnerFlashOn = true;
                captureBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);//mSession
            }else{
                isInnerFlashOn = false;
                captureBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF);
            }

            reader1.setOnImageAvailableListener(mImageAvailable, mBackgroundHandler);

            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    Image image = null;
                    try {
                        image = reader.acquireLatestImage();
                        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                        mImageBytes = new byte[buffer.capacity()];
                        buffer.get(mImageBytes);
                        Log.d("Get Size ", String.valueOf(buffer.capacity()));

                        if(mImageBytes != null && mImageBytes.length >0) {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Bitmap bitmap = BitmapFactory.decodeByteArray(mImageBytes, 0, mImageBytes.length);
                                    if(bitmap.getHeight() < bitmap.getWidth()) bitmap =RotateBitmap(bitmap,90);
                                    isfired = false;
                                    mWarpedImage = findViewById(R.id.RdtWarpImage);
                                    mWarpedImage.setImageBitmap(bitmap);
                                    mWarpedImage.setLayoutParams(mWarpedImage.getLayoutParams());
                                    mWarpedImage.requestLayout();
                                    rdtResults(mImageBytes);
                                    Vibobj.vibrate(50);
                                    progressbar(true);

                                }
                            });
                        }
                        if(isInnerFlashOn) {
                            captureBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);
                            mSession.setRepeatingRequest(captureBuilder.build(), null, null);//mBackgroundHandler
                            mSession.close();
                        }
                        Toast.makeText(MainActivity.this, "Requested for RDT result", Toast.LENGTH_SHORT).show();
                        timeSinceLastChecked = 0;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    finally {
                        if (image != null) {
                            image.close();
                        }
                    }
                }
            };
            reader1.setOnImageAvailableListener(readerListener, mBackgroundHandler);
            final CameraCaptureSession.CaptureCallback captureListener = new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                    isPreviewOff = true;
                    handlerCall = true;

                   }
            };
            mCameraDevice.createCaptureSession(outputSurfaces, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    try {
                        mSession = session;
                        session.capture(captureBuilder.build(), captureListener, mBackgroundHandler);
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }
                @Override
                public void onConfigureFailed(CameraCaptureSession session) {
                }
            }, mBackgroundHandler);

        }catch(Exception e){
            e.printStackTrace();
        }
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

    public void closeCamera() {
        if (null != mCameraDevice) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
        if (null != mImageReader) {
            mImageReader.close();
            mImageReader = null;
        }
    }
    private void startPreview() {
        if (mCameraDevice == null ||!mTextureView.isAvailable() ||mPreviewSize == null || mImageReader == null) {
            return;
        }
        isPreviewOff = false;
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

            mCameraDevice.createCaptureSession(mSurfaces,new CameraCaptureSession.StateCallback() {
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
            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void setUpCaptureRequestBuilder(CaptureRequest.Builder builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
    }

    AcceptanceStatus prevStat;
    long prevTime=0;
    static int counter =0;
    private void repositionRect(AcceptanceStatus status){
        int height = mTextureView.getHeight();
        int width = mTextureView.getWidth();
        mRectView.bringToFront();
        ConstraintLayout.LayoutParams lp = (ConstraintLayout.LayoutParams) mRectView.getLayoutParams();
        lp.width=(int)Math.floor(width*3.0/5);
        lp.height=(int)Math.ceil(height*7.0/9)+1;
        lp.setMargins((int)Math.floor(width/5.0)-1,(int)Math.floor(height/9.0)+1,0,0);
        mRectView.setLayoutParams(lp);
        mRectView.setVisibility(View.VISIBLE);
    }

    @Override
    public void onResume() {
        super.onResume();

        mTextureView.setVisibility(View.VISIBLE);
        mShowImageData = Utils.ApplySettings(this,rdtAPIBuilder,mRdtApi);
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
            isGridDispaly = prefs.getBoolean("gridView",true);

            if(isGridDispaly) {
                gridTable.setVisibility(View.VISIBLE);
            }
            else {
                gridTable.setVisibility(View.INVISIBLE);
            }
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

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
        super.onPause();
        closeCamera();
        stopBackgroundThread();
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

    @Override
    public void processFinish(String output) {
        timeSinceLastChecked=0;
        Continue();
    }
}