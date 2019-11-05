package com.iprd.rdtcamera;

import android.os.Build;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class TestMainRdt {
    public static int index = -1;
    public static void main(String args[]) {
        System.out.println("This is test RDT :");
        String urlString = "http://10.102.10.97:9000/align";
        String metaDataStr = "{\"UUID\":\"a432f9681-a7ff-43f8-a1a6-f777e9362654\",\"Quality_parameters\":{\"brightness\":\"10\"},\"RDT_Type\":\"Flu_Audere\",\"Include_Proof\":\"True\"}";
        String folderPath = "/home/developer/Documents/RdtReader/rdt-reader/RDTTestImage/FluA+B/Afternoon/";
        ArrayList<String> files = getImageList(folderPath);

        postRequest(urlString, metaDataStr, folderPath, files);
    }

    private static void postRequest(String urlString, String metaDataStr, String folderPath, ArrayList<String> files) {
        index = index+1;
        String imgName = files.get(index).substring(files.get(index).lastIndexOf("/")+1);
        File imagefile = new File(files.get(index));//(folderPath + "" + imgName);


        OkHttpClient.Builder b = new OkHttpClient.Builder();
        b.connectTimeout(5,TimeUnit.SECONDS);
        b.readTimeout(30, TimeUnit.SECONDS);
        b.writeTimeout(30, TimeUnit.SECONDS);
        OkHttpClient client =  b.build();

        RequestBody requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("metadata", metaDataStr)
                .addFormDataPart("image", imgName, RequestBody.create(MediaType.parse("image/jpeg"), imagefile))
                .build();

        Request request = new Request.Builder()
                .url(urlString)
                .post(requestBody)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                System.out.println(">>"+e);
                call.cancel();
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                System.out.println(">>"+response.body().string());
                postRequest(urlString, metaDataStr, folderPath, files);
            }
        });
    }

    private static ArrayList<String> getImageList(String folderPath) {
        ArrayList<String> ar = new ArrayList<>();
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Files.walk(Paths.get(folderPath))
                        .filter(path -> Files.isRegularFile(path))
                        .forEach(a->ar.add(a.toString()));
            }else{
                System.out.println("Min SDK should be 26");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ar;
    }
}
