package com.iprd.rdtcamera;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import java.io.IOException;
import java.io.InputStream;

public class ActivityVideo extends AppCompatActivity implements TextureView.SurfaceTextureListener{

    private static final int PICK_VIDEO_REQUEST = 1001;
    private static final String TAG = "SurfaceSwitch";
    private MediaPlayer mMediaPlayer=null;
    private SurfaceHolder mFirstSurface;
    private Uri mVideoUri;
    private ImageView mRectView;
    boolean isMirrored = true;
    public String videoPath = "";
    private RdtAPI mRdtApi;
    byte[] mtfliteBytes = null;
    TextureView mTextureView;
    boolean mStarted=false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_video);
            mRectView = findViewById(R.id.rdtRectVideo);

            mTextureView = (TextureView)findViewById(R.id.textureView);
            mTextureView.setScaleX(1);//isMirrored ? -1 :
            mTextureView.setSurfaceTextureListener(this);

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

    public static String getRealPathFromUri(Context context, Uri contentUri) {
        Cursor cursor = null;
        try {
            String[] proj = {MediaStore.Images.Media.DATA};
            cursor = context.getContentResolver().query(contentUri, proj, null, null, null);
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            return cursor.getString(column_index);
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
    }

    void StartPlayingVideo(SurfaceTexture surface,String videoPath) {
        if(mStarted) return;
        Surface s= new Surface(surface);
        try{
            mMediaPlayer = new MediaPlayer();
            mMediaPlayer.setSurface(s);
            mMediaPlayer.setDataSource(videoPath);
            mMediaPlayer.prepare();
            mMediaPlayer.setOnBufferingUpdateListener(mOnBufferingCB);
            mMediaPlayer.setOnCompletionListener(mOnComplitionCB);
//            mMediaPlayer.setOnPreparedListener(this);
            mMediaPlayer.setOnVideoSizeChangedListener(mOnSizeChangeCB);
            mMediaPlayer.start();
            mStarted = true;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    private MediaPlayer.OnBufferingUpdateListener mOnBufferingCB = new MediaPlayer.OnBufferingUpdateListener() {
        @Override
        public void onBufferingUpdate(MediaPlayer mediaPlayer, int i) {
            Log.d(TAG, "onBufferingUpdate percent:" + i);
        }
    };

    private MediaPlayer.OnCompletionListener mOnComplitionCB = new MediaPlayer.OnCompletionListener(){

        @Override
        public void onCompletion(MediaPlayer mediaPlayer) {
            Log.d(TAG, "onCompletion called");
            releaseMediaPlayer();
        }
    };

    private MediaPlayer.OnVideoSizeChangedListener mOnSizeChangeCB = new MediaPlayer.OnVideoSizeChangedListener() {
        @Override
        public void onVideoSizeChanged(MediaPlayer mediaPlayer, int width, int height) {
            Log.v(TAG, "onVideoSizeChanged called");
            if (width == 0 || height == 0) {
                Log.e(TAG, "invalid video width(" + width + ") or height(" + height + ")");
                return;
            }
            //configureTransform(width,height);
            adjustAspectRatio(width,height);
        }
    };

    private void configureTransform(int viewWidth, int viewHeight) {
        Size mPreviewSize= new Size(1280,720);
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


    public void doStartStop(View view) {
        if (mMediaPlayer == null) {
            Intent pickVideo = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
            pickVideo.setTypeAndNormalize("video/*");
            startActivityForResult(pickVideo, PICK_VIDEO_REQUEST);
        } else {
            releaseMediaPlayer();
        }
    }

    private void releaseMediaPlayer() {
        if(mMediaPlayer!= null) {
            mMediaPlayer.stop();
            mMediaPlayer.release();
            mMediaPlayer = null;
        }
        videoPath = "";
        mStarted=false;
    }

    @Override
    protected void onPause() {
        super.onPause();
        releaseMediaPlayer();
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        releaseMediaPlayer();
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == PICK_VIDEO_REQUEST && resultCode == RESULT_OK) {
            Log.d(TAG, "Got video " + data.getData());
            mVideoUri = data.getData();
            videoPath=getRealPathFromUri(getApplicationContext(),data.getData());
        }
    }

    private void adjustAspectRatio(int videoWidth, int videoHeight) {
        int viewWidth = mTextureView.getWidth();
        int viewHeight = mTextureView.getHeight();
        double aspectRatio = (double) videoHeight / videoWidth;

        int newWidth, newHeight;
        if (viewHeight > (int) (viewWidth * aspectRatio)) {
            // limited by narrow width; restrict height
            newWidth = viewWidth;
            newHeight = (int) (viewWidth * aspectRatio);
        } else {
            // limited by short height; restrict width
            newWidth = (int) (viewHeight / aspectRatio);
            newHeight = viewHeight;
        }
        int xoff = (viewWidth - newWidth) / 2;
        int yoff = (viewHeight - newHeight) / 2;
        Log.v(TAG, "video=" + videoWidth + "x" + videoHeight +
                " view=" + viewWidth + "x" + viewHeight +
                " newView=" + newWidth + "x" + newHeight +
                " off=" + xoff + "," + yoff);

        Matrix txform = new Matrix();
        mTextureView.getTransform(txform);
        txform.setScale((float) newWidth / viewWidth, (float) newHeight / viewHeight);
        //txform.postRotate(10);          // just for fun
        txform.postTranslate(xoff, yoff);
        mTextureView.setTransform(txform);
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.d("1====>>", videoPath);

    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
        return false;
    }
    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
        Log.d("2====>>", videoPath);
    }
    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        Log.d("3====>>", videoPath);
        if(!mStarted){
            StartPlayingVideo(surface,videoPath) ;
        }else{
            if(!mRdtApi.isInProgress()) {
                Bitmap capFrame = mTextureView.getBitmap();
                Process(capFrame);
            }
        }
    }

    private Size getPreviewSize() {
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
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
                ProcessBitmap(capFrame);
//            }
//        }).start();
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
