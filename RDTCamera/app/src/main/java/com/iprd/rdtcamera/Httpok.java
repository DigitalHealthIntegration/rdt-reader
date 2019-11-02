package com.iprd.rdtcamera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.util.Log;

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

    public Httpok(String imgName, byte[] img, String urlString, String metaDataStr){
        this.imgName = imgName;
        this.img = img;
        this.urlString = urlString;
        this.metaDataStr = metaDataStr;
    }

    @Override
    protected String doInBackground(String... strings) {

        try {
            httpOkPostMultipartAndJson();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
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
        System.out.println(">>>>>>>>"+res);
        Bitmap bitmap=null;
        if (response.isSuccessful()) {
            try {
                bitmap = BitmapFactory.decodeStream(response.body().byteStream());
            } catch (Exception e) {
                Log.e("Error", e.getMessage());
                e.printStackTrace();
            }
        }
        if(bitmap != null) Log.i("Maddy",bitmap.getWidth()+"x"+bitmap.getHeight());
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
