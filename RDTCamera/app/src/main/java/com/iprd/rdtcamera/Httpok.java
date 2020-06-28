package com.iprd.rdtcamera;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.AsyncTask;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import static androidx.core.app.NavUtils.navigateUpTo;

public class Httpok extends AsyncTask<String, Void, String> {

    public AsyncResponse delegate = null;

    String imgName;
    byte[] img;
    String urlString;
    String labelName;
    byte[] labels;
    String metaDataStr;
    ProgressBar mProgressBar=null;
    ImageView mImageView=null;
    TextView mResultView=null;
    JSONObject mJsonResult=null;
    Button mGetResult = null;


    static String mHttpURL="http://0.0.0.0:9000/WITS/HIV";
    public void setCtx(Context mCtx) {
        this.mCtx = mCtx;
    }

    Context mCtx=null;

    public Bitmap getResult() {
        return mResult;
    }

    Bitmap mResult=null;
    public Httpok(String imgName, byte[] img, String urlString, String metaDataStr,String labelName,byte[] labels){
        this.imgName = imgName;
        this.img = img;
        this.labelName = labelName;
        this.labels = labels;
        this.urlString = urlString;//"http://3.95.232.90:9000/Quidel/QuickVue/";//urlString;
        this.metaDataStr = metaDataStr;
//        this.mProgressBar= mProgressBar;
//        this.mImageView= view;
        mResult=null;
        mJsonResult=null;
//        mResultView=txtView;
    }

    @Override
    protected String doInBackground(String... strings) {
        try {
            //Thread.sleep(10000);
            httpOkPostMultipartAndJson();
        } catch (IOException e) {
            e.printStackTrace();
        }
//         catch (InterruptedException e) {
//            e.printStackTrace();
//        }
        return null;
    }

    @Override
    protected void onPostExecute(String result) {
//        Log.i("HTTPOK","ONPOSTEXECUTE");
        if (null != mProgressBar) {
            mProgressBar.setVisibility(View.INVISIBLE);
            if (mResult != null) {
                mImageView.setVisibility(View.VISIBLE);
                System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" + mResult);
                mImageView.setImageBitmap(mResult);
                mImageView.bringToFront();
                //Toast.makeText(mCtx,mJsonResult.toString(),Toast.LENGTH_LONG).show();
                if (mJsonResult != null) {
                    String str = "";
                    try {
                        str = mJsonResult.getString("rc") + " " + mJsonResult.getString("msg");
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                    if (null != mResultView) {
                        mResultView.setTextColor(Color.BLACK);
                        mResultView.setText(str);
                        mResultView.setVisibility(View.VISIBLE);
                    }
                    //Toast.makeText(mCtx,str,Toast.LENGTH_LONG).show();
                }
            } else {
                if (mJsonResult != null) {
                    String str = "";
                    try {
                        str = mJsonResult.getString("rc") + " " + mJsonResult.getString("msg");
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                    if (null != mResultView) {
                        mResultView.setTextColor(Color.BLACK);
                        mResultView.setText(str);
                        mResultView.setVisibility(View.VISIBLE);
                    }
                    //Toast.makeText(mCtx,str,Toast.LENGTH_LONG).show();
                }/*else{
                    mResultView.setVisibility(View.INVISIBLE);
                    //mResultView.setText("waiting...");
                }*/
            }


        }
        delegate.processFinish("done");

    }
    @Override
    protected void onPreExecute() {
//        Log.i("HTTPOK","ONPREEXECUTE");
        if(null != mProgressBar  && mProgressBar.getVisibility() == View.INVISIBLE){
            mProgressBar.setVisibility(View.VISIBLE);
            mProgressBar.bringToFront();
        }
    }

    private void httpOkPostMultipartAndJson() throws IOException {
        OkHttpClient.Builder b = new OkHttpClient.Builder();
        b.connectTimeout(5, TimeUnit.SECONDS);
        b.readTimeout(30, TimeUnit.SECONDS);
        b.writeTimeout(30, TimeUnit.SECONDS);
        OkHttpClient client =  b.build();

        RequestBody requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("metadata", metaDataStr)
                .addFormDataPart("image", imgName, RequestBody.create(MediaType.parse("image/jpeg"), img))
                .addFormDataPart("label",labelName,RequestBody.create(MediaType.parse("text/json"),labels))
                .build();

        Request request = new Request.Builder()
                .url(urlString)
                .post(requestBody)
                .build();

        Response response = client.newCall(request).execute();
        String res = response.body().string();
        //System.out.println(">>>>>>>>"+res);
        Bitmap bitmap=null;
        if (response.isSuccessful()) {
            try {
                String[] s = res.split("Content-Type:");
                for (String a : s) {
                    if(a.contains("image/jpeg\r\n\r\n")) {
                        String[] i=a.split("image/jpeg\r\n\r\n");
                        if(i.length >1){
                            String[] k=i[1].split("--");
                            System.out.println(k[0]);
                            byte[] decodedString = Base64.decode(k[0], Base64.DEFAULT);
                            mResult = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
                        }
                    }
                    if(a.contains("application/json\r\n\r\n")) {
                        String[] i=a.split("application/json\r\n\r\n");
                        if(i.length >1){
                            JSONObject obj = new JSONObject(i[1]);
                            Log.i("UUID",obj.getString("UUID"));
//                            Log.i("msg",obj.getString("msg"));
                            Log.i("rc",obj.getString("rc"));
                            mJsonResult = obj;

                        }
                    }
                }
            } catch (Exception e) {
                Log.e("Error", e.getMessage());
                e.printStackTrace();
            }
        }
    }
  }
