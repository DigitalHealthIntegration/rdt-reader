package com.iprd.rdtcamera;

import android.app.Activity;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;



import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageView;

import org.opencv.core.Point;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class ActivityVideo extends AppCompatActivity implements TextureView.SurfaceTextureListener{

    private static final int PICK_VIDEO_REQUEST = 1001;
    private static final String TAG = "SurfaceSwitch";
    private MediaPlayer mMediaPlayer;
    private SurfaceHolder mFirstSurface;
    private Uri mVideoUri;
    private ImageView mRectView;
    boolean isMirrored = true;
    public String videoPath = "";
    MediaPlayer mediaPlayer = null;
    private RdtAPI mRdtApi;
    byte[] mtfliteBytes = null;
    TextureView textureView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_video);
            mRectView = findViewById(R.id.rdtRectVideo);

            textureView = (TextureView)findViewById(R.id.textureView);
            textureView.setSurfaceTextureListener(this);
            textureView.setScaleX(1);//isMirrored ? -1 : 

            Config c = new Config();
            try {
                c.mTfliteB = ReadAssests();
            } catch (IOException e) {
                e.printStackTrace();
            }
            mRdtApi = new RdtAPI();
            mRdtApi.init(c);
        }
    }

    public void doStartStop(View view) {
        if (mMediaPlayer == null) {
            Intent pickVideo = new Intent(Intent.ACTION_PICK);
            pickVideo.setTypeAndNormalize("video/*");
            startActivityForResult(pickVideo, PICK_VIDEO_REQUEST);
        } else {
            mMediaPlayer.stop();
            mMediaPlayer.release();
            mMediaPlayer = null;
            videoPath = "";
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == PICK_VIDEO_REQUEST && resultCode == RESULT_OK) {
            Log.d(TAG, "Got video " + data.getData());
            mVideoUri = data.getData();
            videoPath = data.getData().getPath();
        }
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.d("1====>>", videoPath);
    }
    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {return false;}
    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
        Log.d("2====>>", videoPath);
    }
    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
        Log.d("3====>>", videoPath);
        videoPath = videoPath.replace("external_files","sdcard");
        if(mediaPlayer == null) {

            mediaPlayer = new MediaPlayer();

            mediaPlayer.setSurface(new Surface(surfaceTexture));
            if (videoPath.length() > 1)
                try {
                    /*Matrix txform = new Matrix();
                    textureView.getTransform(txform);
                    txform.setScale((float) newWidth / viewWidth, (float) newHeight / viewHeight);
                    txform.postTranslate(xoff, yoff);
                    textureView.setTransform(txform);*/

                    mediaPlayer.setDataSource(videoPath);
                    mediaPlayer.prepare();
                    mediaPlayer.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
        }else{
            Log.d(".... ","onSurfaceTextureUpdated");
            if(!mRdtApi.isInProgress()) {
                Bitmap capFrame = textureView.getBitmap();
                Process(capFrame);
            }
        }
    }

    /*private void configureTransform() {

        mPreviewSize = getPreviewSize();
        if (null == textureView || null == mPreviewSize || null == this) {
            return;
        }

        int viewWidth = textureView.getWidth();
        int viewHeight = textureView.getHeight();

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
        textureView.setTransform(matrix);
    }*/

    private Size getPreviewSize() {

        //


        //
        return null;
    }

    byte[] ReadAssests() throws IOException {
        InputStream is=getAssets().open("tflite.lite");
        mtfliteBytes=new byte[is.available()];
        is.read( mtfliteBytes);
        is.close();
        return mtfliteBytes;
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
       {
            status.mSharpness = mRdtApi.mSharpness;
            status.mBrightness = mRdtApi.mBrightness;
        }
        long et = System.currentTimeMillis()-st;
        Log.i("Total Processing Time "," "+ et);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                repositionRect(status);
                Log.d("~~~~~~~~~~~","~~~~~~~");
            }
            long prevTime=0;
            AcceptanceStatus prevStat;
            private void repositionRect(AcceptanceStatus status) {

                if(mRectView == null)return;
                if(status.mRDTFound){
                    prevTime = System.currentTimeMillis();
                    prevStat = status;
                }else {
                    long curr = System.currentTimeMillis();
                    mRectView.setVisibility(View.INVISIBLE);
                    //rdtDataToBeDisplay.setVisibility(View.INVISIBLE);
                    return;
                }

                mRectView.bringToFront();
                //rdtDataToBeDisplay.bringToFront();
               ConstraintLayout.LayoutParams lp = (ConstraintLayout.LayoutParams) mRectView.getLayoutParams();

                lp.width = prevStat.mBoundingBoxWidth;
                lp.height=prevStat.mBoundingBoxHeight;
                lp.setMargins(prevStat.mBoundingBoxX,status.mBoundingBoxY,0,0);
                //Log.d("Box","Bounds "+lp.width+"x"+lp.height+" Position "+boxPosition.getWidth()+"x"+boxPosition.getHeight());
                mRectView.setLayoutParams(lp);
                mRectView.setVisibility(View.VISIBLE);


            }
        });
    }

}
