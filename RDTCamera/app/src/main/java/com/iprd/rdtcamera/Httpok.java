package com.iprd.rdtcamera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class Httpok extends AsyncTask<String, Void, String> {
    String imgName;
    byte[] img;
    String urlString;
    String metaDataStr;
    ProgressBar mProgressBar=null;
    ImageView mImageView=null;
    JSONObject mJsonResult=null;

    public void setCtx(Context mCtx) {
        this.mCtx = mCtx;
    }

    Context mCtx=null;

    public Bitmap getResult() {
        return mResult;
    }

    Bitmap mResult=null;
    public Httpok(String imgName, byte[] img, String urlString, String metaDataStr, ProgressBar mProgressBar,ImageView view){
        this.imgName = imgName;
        this.img = img;
        this.urlString = urlString;
        this.metaDataStr = metaDataStr;
        this.mProgressBar= mProgressBar;
        this.mImageView= view;
        mResult=null;
        mJsonResult=null;
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
        if(null != mProgressBar){
            mProgressBar.setVisibility(View.INVISIBLE);
            if(mResult != null){
                mImageView.setVisibility(View.VISIBLE);
                mImageView.setImageBitmap(mResult);
                mImageView.bringToFront();
                Toast.makeText(mCtx,mJsonResult.toString(),Toast.LENGTH_LONG).show();
            }
        }
    }
    @Override
    protected void onPreExecute() {
//        Log.i("HTTPOK","ONPREEXECUTE");
        if(null != mProgressBar){
            mProgressBar.setVisibility(View.VISIBLE);
            mProgressBar.bringToFront();
        }
    }

    private void httpOkPostMultipartAndJson() throws IOException {
        OkHttpClient client = new OkHttpClient();
        //File imagefile = new File(folderPath+""+imgName);
        RequestBody requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("metadata", metaDataStr)
                .addFormDataPart("image", imgName, RequestBody.create(MediaType.parse("image/jpeg"), img))
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
                            Log.i("msg",obj.getString("msg"));
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

/*
 String folderPath = "/sdcard/aa/mgd/";
        String imgName = "KH5.jpg";
        String urlString = "http://10.102.10.97:9000/align";
        String metaDataStr = "{\"UUID\":\"a432f9681-a7ff-43f8-a1a6-f777e9362654\",\"Quality_parameters\":{\"brightness\":\"10\"},\"RDT_Type\":\"Flu_Audere\",\"Include_Proof\":\"True\"}";
        try{
            Okhttp mr = new Okhttp(folderPath, imgName, urlString, metaDataStr);
            mr.execute();
        }catch(Exception ex){
            ex.printStackTrace();
        }
 */
