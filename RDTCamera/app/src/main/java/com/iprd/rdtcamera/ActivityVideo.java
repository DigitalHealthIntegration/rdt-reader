package com.iprd.rdtcamera;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.jcodec.api.FrameGrab;
import org.jcodec.api.JCodecException;
import org.jcodec.common.AndroidUtil;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.common.model.Picture;
import org.jcodec.scale.BitmapUtil;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;

import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;
import static android.Manifest.permission_group.CAMERA;

public class ActivityVideo extends AppCompatActivity {
    private static final int PICK_VIDEO_REQUEST = 1001;
    private static final String mModelFileName="tflite.lite";
    static String TAG = ActivityVideo.class.getName();
    enum PlayPause {PLAY, PAUSE};
    Uri mVideoUri;
    String videoPath;
    Button mSelectVideo;
    Button mPlayPause;
    Button mGetResult;
    ImageView mShowImage,mRdtImage;
    TextView mResultView;
    Button preferenceSettingBtn;
    private RdtAPI.RdtAPIBuilder rdtAPIBuilder;
    private RdtAPI mRdtApi;
    Short mShowImageData;
    Bitmap mCapFrame;
    ProgressBar mCyclicProgressBar;
    SharedPreferences prefs;

    PlayPause mState = PlayPause.PAUSE;
    boolean mRunningloop = false;

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
        setContentView(R.layout.activity_video);
        mCyclicProgressBar = findViewById(R.id.loader);
        mRdtImage = findViewById(R.id.rdt);
        mGetResult = findViewById(R.id.getResult);
        mResultView = findViewById(R.id.ResultView);
        preferenceSettingBtn = (Button) findViewById(R.id.preferenceSettingBtn);
        preferenceSettingBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(ActivityVideo.this, MyPreferencesActivity.class);
                startActivity(i);
            }
        });
        prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

        MappedByteBuffer mMappedByteBuffer = null;
        try {
            mMappedByteBuffer = Utils.loadModelFile(getAssets(), mModelFileName);
        } catch (Exception e) {
            e.printStackTrace();
        }

        rdtAPIBuilder = new RdtAPI.RdtAPIBuilder();
        rdtAPIBuilder = rdtAPIBuilder.setModel(mMappedByteBuffer);
        mShowImageData = Utils.ApplySettings(this, rdtAPIBuilder, null);
        mRdtApi = rdtAPIBuilder.build();

        //call the setter for saving functions
        Utils.ApplySettings(this, null, mRdtApi);
        mRdtApi.setmPlaybackMode(true);

        mShowImage = (ImageView) findViewById(R.id.ShowImage);
        mSelectVideo = (Button) findViewById(R.id.SelectFile);
        mSelectVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mRdtImage.setVisibility(View.INVISIBLE);
                mCyclicProgressBar.setVisibility(View.INVISIBLE);
                mGetResult.setVisibility(View.INVISIBLE);
                mResultView.setVisibility(View.INVISIBLE);
                mRunningloop = false;
                mPlayPause.setText("");
                try {
                    Thread.sleep(200);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                SelectVideo();
            }
        });

        mGetResult.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mCyclicProgressBar.setVisibility(View.VISIBLE);
                mCyclicProgressBar.bringToFront();
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                mCapFrame.compress(Bitmap.CompressFormat.JPEG, 90, stream);
                byte[] byteArray = stream.toByteArray();

                String urlString = prefs.getString("rdtCheckUrl","http://3.82.11.139:9000/align");
                String guid = String.valueOf(java.util.UUID.randomUUID());
                String metaDataStr = "{\"UUID\":" +"\"" + guid +"\",\"Quality_parameters\":{\"brightness\":\"10\"},\"RDT_Type\":\"Flu_Audere\",\"Include_Proof\":\"True\"}";
                try{
                    Httpok mr = new Httpok("img.jpg",byteArray, urlString, metaDataStr,mCyclicProgressBar,mRdtImage,mResultView);
                    mr.setCtx(getApplicationContext());
                    mr.execute();
                }catch(Exception ex){
                    ex.printStackTrace();
                }
            }
        });
        mPlayPause = (Button) findViewById(R.id.PlayPause);
        mPlayPause.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
            if (mState == PlayPause.PLAY) {
                mState = PlayPause.PAUSE;
                setFilePickerVisibility(true);
                mPlayPause.setText("PAUSE");
                mGetResult.setVisibility(View.VISIBLE);
            } else if (mRunningloop) {
                mState = PlayPause.PLAY;
                setFilePickerVisibility(false);
                mPlayPause.setText("");
                mResultView.setVisibility(View.INVISIBLE);
                mGetResult.setVisibility(View.INVISIBLE);
                mRdtImage.setVisibility(View.INVISIBLE);
                mCyclicProgressBar.setVisibility(View.INVISIBLE);
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
                mSelectVideo.setVisibility(vis ? View.VISIBLE : View.INVISIBLE);
            }
        });
    }

    private void playVideo(String videoFilename) {
        try {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(getApplicationContext(), "Tap on centre of screen to pause video", Toast.LENGTH_LONG).show();
                }
            });
            setFilePickerVisibility(false);
            mRunningloop = true;
            mState = PlayPause.PLAY;
            File file = new File(videoFilename);
            FrameGrab grab = FrameGrab.createFrameGrab(NIOUtils.readableChannel(file));

            boolean process = true;
            int count = 0;
            while (mRunningloop) {
                if (mState == PlayPause.PAUSE) {
                    Thread.sleep(100);
                    continue;
                }
                Picture picture = grab.getNativeFrame();
                if (picture == null) {
                    break;
                } else {
                    System.out.println(picture.getWidth() + "x" + picture.getHeight() + " " + picture.getColor());
                    long st = System.currentTimeMillis();

                    mCapFrame = AndroidUtil.toBitmap(picture);
                    Log.i("IPRD", "frame" + count++ + "_" + mCapFrame.getWidth() + "x" + mCapFrame.getHeight());
                    long et = System.currentTimeMillis()- st;
                    Log.i("Bitmap Conversion Time "," "+ et);

                    st = System.currentTimeMillis();
                    final AcceptanceStatus status = mRdtApi.checkFrame(mCapFrame);
                    et = System.currentTimeMillis()- st;

                    Log.i("Pre Divide Time ",""+ mRdtApi.getDivideTime());
                    Log.i("TfLite Time ",""+ mRdtApi.getTfliteTime());
                    Log.i("ROI Finding Time ",""+ mRdtApi.getROIFindingTime());

                    Log.i("Pre Processing Time ",""+mRdtApi.getPreProcessingTime());
                    Log.i("TF Processing Time "," "+ mRdtApi.getTensorFlowProcessTime());
                    Log.i("Post Processing Time "," "+ mRdtApi.getPostProcessingTime());
                    Log.i("Total Processing Time "," "+ et);
                    String frNoSB = "F["+count + "]\nS[" + mRdtApi.getSharpness() + "]\n" + "B[" + mRdtApi.getBrightness() + "]";
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
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JCodecException e) {
            e.printStackTrace();
        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            setFilePickerVisibility(true);
            mState = PlayPause.PAUSE;
            mRunningloop = false;
        }
    }
}
