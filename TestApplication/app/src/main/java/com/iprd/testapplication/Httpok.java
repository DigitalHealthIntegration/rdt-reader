package com.iprd.testapplication;

import android.os.AsyncTask;

import java.io.File;
import java.io.IOException;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class Httpok extends AsyncTask<String, Void, String> {
    String folderPath;
    String imgName;
    String urlString;
    String metaDataStr;

    public Httpok(String folderPath, String imgName, String urlString, String metaDataStr){
        this.folderPath = folderPath;
        this.imgName = imgName;
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

        File imagefile = new File(folderPath+""+imgName);
        RequestBody requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("metadata", metaDataStr)
                .addFormDataPart("image", imgName, RequestBody.create(MediaType.parse("image/jpeg"), imagefile))
                .build();

        Request request = new Request.Builder()
                .url(urlString)
                .post(requestBody)
                .build();

        Response response = client.newCall(request).execute();
        System.out.println(">>>>>>>>"+response.toString());
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
