package com.iprd.rdtcamera;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;



import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.util.Log;
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
            textureView.setScaleX(isMirrored ? -1 : 1);
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
