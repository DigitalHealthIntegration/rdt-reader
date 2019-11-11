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
import android.hardware.camera2.TotalCaptureResult;
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
import android.widget.ProgressBar;
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

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.Manifest.permission_group.CAMERA;
import static com.iprd.rdtcamera.ModelInfo.mModelFileName;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.floodFill;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "AndroidCameraApi";
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private ImageView mRectView;

    private ImageView disRdtResultImage;
    Button mGetResult;
    Button startBtn;
    TextView mResultView;
    Boolean isPreviewOff = false;
    Boolean shouldOffTorch = false;

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    SharedPreferences prefs;
    ProgressBar mCyclicProgressBar;

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

    private short mShowImageData=0;
    public Switch mode;
    public Switch torch;
    public Switch saveData;

    int idx;
    RdtAPI mRdtApi=null;
    private RdtAPI.RdtAPIBuilder rdtAPIBuilder=null;

    private boolean checkpermission(){
        System.out.println("..>>"+ WRITE_EXTERNAL_STORAGE);
        int res  = ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA);
        int res1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int res2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        return res1 == PackageManager.PERMISSION_GRANTED && res == PackageManager.PERMISSION_GRANTED && res2 == PackageManager.PERMISSION_GRANTED;
    }
    private void requestPermission(){
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

        mGetResult = findViewById(R.id.getResult);
        prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

        mRectView = findViewById(R.id.rdtRect);
        disRdtResultImage = findViewById(R.id.disRdtResultImage);
        rdtDataToBeDisplay = findViewById(R.id.rdtDataToBeDisplay);
        mCyclicProgressBar = findViewById(R.id.loader);

        mResultView = findViewById(R.id.ResultView);
        // mCyclicProgressBar.setVisibility(View.INVISIBLE);
        startBtn = findViewById(R.id.startBtn);
        //rdtDataToBeDisplay.setTextColor(0x000000FF);
        // preferences
        preferenceSettingBtn = (Button) findViewById(R.id.preferenceSettingBtn);
        preferenceSettingBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(MainActivity.this, MyPreferencesActivity.class);
                startActivity(i);
            }
        });
        byte[] mTfliteB=null;
        MappedByteBuffer mMappedByteBuffer=null;
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
        mRdtApi.setRotation(true);

        //mRdtApi.setSaveImages(true);
//        mRdtApi.saveInput(true);
//        mRdtApi.setSavePoints(true);

        //call the setter for saving functions
        Utils.ApplySettings(this,null,mRdtApi);
        /// Set Torch button
        torch = (Switch) findViewById(R.id.torch);
        //torch.setVisibility(View.VISIBLE);
        torch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                try{
                    if (shouldOffTorch == false){
                        if (isChecked) {
                            // The toggle is enabled
                            mPreviewBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);
                            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, null);
                        } else {
                            // The toggle is disabled
                            mPreviewBuilder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF);
                            mPreviewSession.setRepeatingRequest(mPreviewBuilder.build(), null, null);
                        }
                    }else{
                        shouldOffTorch =false;
                    }
                }catch (CameraAccessException e){
                    e.printStackTrace();
                }
            }
        });

        /// Set Save button
        saveData = (Switch) findViewById(R.id.saveData);
        //saveData.setVisibility(View.VISIBLE);
        saveData.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mRdtApi.setSaveImages(true);
                } else {
                    mRdtApi.setSaveImages(false);
                }
            }
        });

        ///Video Play
        mode = (Switch) findViewById(R.id.mode);
        mode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(isPreviewOff){
                    startPreview();
                    if(mCyclicProgressBar.getVisibility() == View.VISIBLE) {
                        mCyclicProgressBar.setVisibility(View.INVISIBLE);
                        mResultView.setVisibility(View.INVISIBLE);
                        mGetResult.setVisibility(View.VISIBLE);
                        startBtn.setVisibility(View.INVISIBLE);
                    }

                }//?
                if (isChecked) {
                    mode.setChecked(false);
                    mode.setChecked(false);
                    torch.setChecked(false);

                    Intent i = new Intent(MainActivity.this, ActivityVideo.class);
                    i.putExtra("videoPath","aaaaaa");
                    i.setFlags(Intent.FLAG_ACTIVITY_FORWARD_RESULT);
                    startActivity(i);
                } else {
                    Log.d(">>Mode Switch<<","OFF");
                }
            }
        });
        mGetResult.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
               /* if(mCyclicProgressBar.getVisibility() == View.INVISIBLE) {
                    mCyclicProgressBar.setVisibility(View.VISIBLE);
                    mCyclicProgressBar.bringToFront();
                }*/
                //progressbar(true);
                getRDTResultData();
                startBtn.setVisibility(View.VISIBLE);
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

                startPreview();
                mResultView.setVisibility(View.INVISIBLE);
                mGetResult.setVisibility(View.VISIBLE);
                startBtn.setVisibility(View.INVISIBLE);
                /*if(mCyclicProgressBar.getVisibility() == View.VISIBLE) {
                    mCyclicProgressBar.setVisibility(View.INVISIBLE);
                }*/
//                disRdtResultImage = null;
//                disRdtResultImage = findViewById(R.id.disRdtResultImage);
                progressbar(false);
                disRdtResultImage.setVisibility(View.INVISIBLE);
            }
        });
    }

    void progressbar(boolean isVisible){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mCyclicProgressBar.setVisibility(isVisible?View.VISIBLE:View.INVISIBLE);
            }
        });
    }

    byte[] ReadAssests() throws IOException {
        byte[] mtfliteBytes=null;
        InputStream is=getAssets().open(mModelFileName);
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
            Image image = null;
            try {
                image = reader.acquireLatestImage();
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
            //Log.d(".... ","onSurfaceTextureUpdated");
            mode.setVisibility(View.VISIBLE);
            saveData.setVisibility(View.VISIBLE);
            torch.setVisibility(View.VISIBLE);

            if(!mRdtApi.isInprogress()) {
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
            final AcceptanceStatus status = mRdtApi.checkFrame(capFrame);
            if(mShowImageData != 0){
                status.mSharpness = mRdtApi.getSharpness();
                status.mBrightness = mRdtApi.getBrightness();
            }
            long et = System.currentTimeMillis()-st;
            Log.i("BBF",status.mBoundingBoxX+"x"+status.mBoundingBoxY+"-"+status.mBoundingBoxWidth+"x"+status.mBoundingBoxHeight);

//            Log.i("Pre Processing Time ",""+mRdtApi.getPreProcessingTime());
//            Log.i("TF Processing Time "," "+ mRdtApi.getTensorFlowProcessTime());
//            Log.i("Post Processing Time "," "+ mRdtApi.getPostProcessingTime());
 //           Log.i("Total Processing Time "," "+ et);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    repositionRect(status);
                }
            });
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
            Log.d("Camera ",size.toString());
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

    private void getRDTResultData(){
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try{
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

            ImageReader reader = ImageReader.newInstance(width, height, ImageFormat.JPEG, 1);
            List<Surface> outputSurfaces = new ArrayList<Surface>(2);
            outputSurfaces.add(reader.getSurface());
            outputSurfaces.add(new Surface(mTextureView.getSurfaceTexture()));
            final CaptureRequest.Builder captureBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureBuilder.set(CaptureRequest.JPEG_QUALITY, (byte) 90);
            captureBuilder.addTarget(reader.getSurface());
            captureBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(rotation));

            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    Thread.yield();
                    Image image = null;
                    try {
                        progressbar(true);
                        image = reader.acquireLatestImage();
                        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                        byte[] bytes = new byte[buffer.capacity()];
                        buffer.get(bytes);
                        rdtResults(bytes);
                        Toast.makeText(MainActivity.this, "Requested for RDT result", Toast.LENGTH_SHORT).show();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    } finally {
                        if (image != null) {
                            image.close();
                        }
                    }
                }

                private void rdtResults(byte[] bytes) throws IOException {
                    OutputStream output = null;
                    try {
                        String urlString = prefs.getString("rdtCheckUrl","http://34.204.97.37:9000/Quidel/QuickVue");
                        System.out.println(">>>>>>>>"+urlString);
                        String guid = String.valueOf(java.util.UUID.randomUUID());
                        String metaDataStr = "{\"UUID\":" +"\"" + guid +"\",\"Quality_parameters\":{\"brightness\":\"10\"},\"RDT_Type\":\"Flu_Audere\",\"Include_Proof\":\"True\"}";
                        try{
                            Httpok mr = new Httpok("img.jpg",bytes, urlString, metaDataStr,mCyclicProgressBar,disRdtResultImage,mResultView);

                            mr.setCtx(getApplicationContext());
                            mr.execute();
                        }catch(Exception ex){
                            ex.printStackTrace();
                        }
                    } finally {
                        if (null != output) {
                            output.close();
                        }
                    }
                }
            };

            reader.setOnImageAvailableListener(readerListener, mBackgroundHandler);
            final CameraCaptureSession.CaptureCallback captureListener = new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                    isPreviewOff = true;
                    /*try {
                        //Thread.sleep(500);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }*/
                }
            };
            mCameraDevice.createCaptureSession(outputSurfaces, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    try {
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
            System.out.println(">>>>>>>>>>>"+e);
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

    private void closeCamera() {
        closePreviewSession();
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
    private void repositionRect(AcceptanceStatus status){

        if(mRectView == null)return;
        if(disRdtResultImage == null) return;
        if(status.mRDTFound){
            prevTime = System.currentTimeMillis();
            prevStat = status;
        }else {
            long curr = System.currentTimeMillis();
            mRectView.setVisibility(View.INVISIBLE);
            if(rdtDataToBeDisplay != null) {
                rdtDataToBeDisplay.setVisibility(View.INVISIBLE);
            }
            return;
        }

        mRectView.bringToFront();
        rdtDataToBeDisplay.bringToFront();
        Point boundsRatio = new Point(prevStat.mBoundingBoxWidth*1.0/CAMERA2_PREVIEW_SIZE.getWidth(),prevStat.mBoundingBoxHeight*1.0/CAMERA2_PREVIEW_SIZE.getHeight()),
                positionRatio = new Point(prevStat.mBoundingBoxX*1.0/CAMERA2_PREVIEW_SIZE.getWidth(),prevStat.mBoundingBoxY*1.0/CAMERA2_PREVIEW_SIZE.getHeight());

        ConstraintLayout.LayoutParams lp = (ConstraintLayout.LayoutParams) mRectView.getLayoutParams();
        lp.width = prevStat.mBoundingBoxWidth;
        lp.height=prevStat.mBoundingBoxHeight;
        lp.setMargins(prevStat.mBoundingBoxX,status.mBoundingBoxY,0,0);
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