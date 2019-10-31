package com.iprd.testapplication;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;

import java.nio.MappedByteBuffer;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.Manifest.permission_group.CAMERA;

public class MainActivity extends AppCompatActivity {
    private static final int PICK_VIDEO_REQUEST = 1001;
    static String TAG = MainActivity.class.getName();
    enum PlayPause {PLAY,PAUSE};
    Uri mVideoUri;
    String videoPath;
    Button mSelectVideo;
    Button mPlayPause;
    ImageView mShowImage;
    Button preferenceSettingBtn;
    private RdtAPI.RdtAPIBuilder rdtAPIBuilder;
    private RdtAPI mRdtApi;
    Short mShowImageData;
    Bitmap mCapFrame;

    PlayPause mState=PlayPause.PAUSE;
    boolean mRunningloop=false;
    private boolean checkpermission() {
        System.out.println("..>>" + WRITE_EXTERNAL_STORAGE);
        int res = ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA);
        int res1 = ContextCompat.checkSelfPermission(getApplicationContext(), WRITE_EXTERNAL_STORAGE);
        int res2 = ContextCompat.checkSelfPermission(getApplicationContext(), READ_EXTERNAL_STORAGE);
        return res1 == PackageManager.PERMISSION_GRANTED && res == PackageManager.PERMISSION_GRANTED && res2 == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, CAMERA, Manifest.permission.CAMERA}, 200);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (!checkpermission()) {
            requestPermission();
        }
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        preferenceSettingBtn = (Button) findViewById(R.id.preferenceSettingBtn);
        preferenceSettingBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(MainActivity.this, MyPreferencesActivity.class);
                startActivity(i);
            }
        });
        MappedByteBuffer mMappedByteBuffer = null;
        try {
            mMappedByteBuffer = Utils.loadModelFile(getAssets(), "tflite.lite");
        } catch (Exception e) {
            e.printStackTrace();
        }
        rdtAPIBuilder = new RdtAPI.RdtAPIBuilder();
        rdtAPIBuilder = rdtAPIBuilder.setModel(mMappedByteBuffer);
        mShowImageData = Utils.ApplySettings(this, rdtAPIBuilder, null);
        mRdtApi = rdtAPIBuilder.build();

        //call the setter for saving functions
        Utils.ApplySettings(this, null, mRdtApi);

        mShowImage = (ImageView) findViewById(R.id.ShowImage);
        mSelectVideo = (Button) findViewById(R.id.SelectFile);
        mSelectVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mRunningloop = false;
                mPlayPause.setText("");
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                SelectVideo();
            }
        });
        mPlayPause = (Button) findViewById(R.id.PlayPause);
        mPlayPause.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
               if(mState == PlayPause.PLAY) {
                   mState = PlayPause.PAUSE;
                   setFilePickerVisibility(true);
                   mPlayPause.setText("PAUSE");
               }else if(mRunningloop){
                   mState = PlayPause.PLAY;
                   setFilePickerVisibility(false);
                   mPlayPause.setText("");
               }
            }
        });
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
    @Override
    public void onResume() {
        super.onResume();
        mShowImageData = Utils.ApplySettings(this, rdtAPIBuilder, mRdtApi);
    }
    private static final int FILE_SELECT_CODE = 0;

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_VIDEO_REQUEST && resultCode == RESULT_OK) {
            Log.i(TAG, "Got video " + data.getData());
            mVideoUri = data.getData();
            final String videoPath = getRealPathFromUri(getApplicationContext(), data.getData());
            Log.i(TAG, "Got video path" + videoPath);
            if (videoPath != null) {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        playVideo(videoPath);
                    }
                }).start();
            }
        }
    }

    void SelectVideo() {
        Intent pickVideo = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        pickVideo.setType("video/mp4");
        startActivityForResult(pickVideo, PICK_VIDEO_REQUEST);
    }

    void setFilePickerVisibility(final boolean vis) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
            mSelectVideo.setVisibility(vis?View.VISIBLE:View.INVISIBLE);
            }
        });
    }

    private void playVideo(String videoFilename) {
        try {
            setFilePickerVisibility(false);
            mRunningloop = true;
            mState = PlayPause.PLAY;
            final FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFilename);
            AndroidFrameConverter converter = new AndroidFrameConverter();
            grabber.start();
            boolean process = true;
            int count = 0;
            while (mRunningloop) {
                if (mState == PlayPause.PAUSE) {
                    Thread.sleep(1000);
                    continue;
                }
                Frame frame = null;
                frame = grabber.grab();
                if (frame == null) {
                    break;
                }
                if (frame.image != null) {
                    mCapFrame = converter.convert(frame);
                    Log.i("Madhav", "frame" + count++ + "_" + mCapFrame.getWidth() + "x" + mCapFrame.getHeight());
                    final AcceptanceStatus status = mRdtApi.checkFrame(mCapFrame);
                    String frNoSB = count+" S["+mRdtApi.getSharpness() +"]" + "B[" + mRdtApi.getBrightness() +"]";
                    mRdtApi.SetText(frNoSB, status);
                    final Bitmap ret = mRdtApi.getLocalcopyAsBitmap();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                        mShowImage.setImageBitmap(ret);
                        }
                    });
                }
            }
            Log.i("Madhav", "Done");
        } catch (Exception ex) {
            ex.printStackTrace();
        }finally {
            setFilePickerVisibility(true);
            mState = PlayPause.PAUSE;
            mRunningloop = false;
        }
    }
}
