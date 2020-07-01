package com.iprd.rdtcamera;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;

import static android.os.SystemClock.sleep;
import static com.iprd.rdtcamera.Httpok.mHttpURL;

public class LabelScreen extends AppCompatActivity implements AsyncResponse {
    private ImageView clickedPicture;
    private Button submit;
    private Button back;
    private RadioGroup radioGroup;
    private RadioButton selectedButton;
    private TextView notes;
    private RadioButton negative;
    private Bitmap recievedImage;
    private Context mctx;
    private SharedPreferences prefs;
    TextView serverConnectionText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mctx = getApplicationContext();
        setContentView(R.layout.activity_label_screen);
        clickedPicture = (ImageView) findViewById(R.id.rdtImage);
        back = (Button) findViewById(R.id.back);
        submit = (Button) findViewById(R.id.submit);
        serverConnectionText = findViewById(R.id.serverConnection);

        radioGroup = (RadioGroup) findViewById(R.id.radioGroup);
        notes = (TextView) findViewById(R.id.notes);
        prefs = this.getSharedPreferences("MyPrefsFile", MODE_PRIVATE);//PreferenceManager.getDefaultSharedPreferences(c);

        if(getIntent().hasExtra("image_path")) {
            recievedImage = readFile(String.valueOf(getIntent().getStringExtra("image_path")));
            clickedPicture.setImageBitmap(recievedImage);
        }
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent nextIntent = new Intent(mctx, MainActivity.class);
                nextIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);

                startActivity(nextIntent);
                finish();
            }
        });
        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String tmpText = notes.getText().toString();
                try{
                    if(tmpText.equals("")){
                        throw  new NullPointerException("Need to add note");
                    }
                    if(isNetworkAvailable()){
                        ByteArrayOutputStream stream = new ByteArrayOutputStream();
                        recievedImage.compress(Bitmap.CompressFormat.PNG, 100, stream);
                        byte[] byteArray = stream.toByteArray();
                        int selectedId = radioGroup.getCheckedRadioButtonId();
                        selectedButton = (RadioButton) findViewById(selectedId);
                        JSONObject labelResult = new JSONObject();
                        try {
                            labelResult.put("notes",tmpText);
                            labelResult.put("label",selectedButton.getText());

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                        rdtResults(byteArray,labelResult.toString().getBytes());
                        Intent nextIntent = new Intent(mctx, MainActivity.class);
                        nextIntent.putExtra("submitToserver",true);
                        startActivity(nextIntent);


                    }
                    else throw new ConnectException("no internet connection");

                }
                catch (NullPointerException e){
                    e.printStackTrace();
                    Toast.makeText(LabelScreen.this,"Please select an option and write the RDT note before submitting",Toast.LENGTH_LONG).show();
                }
                catch(ConnectException e){
                    e.printStackTrace();
                    Toast.makeText(LabelScreen.this,"Please check internet connectivity",Toast.LENGTH_LONG).show();
                }

            }
        });
    }
    private boolean isNetworkAvailable() {
        ConnectivityManager connectivityManager
                = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        return activeNetworkInfo != null;
    }



    public void rdtResults(byte[] imagebytes,byte [] labelByte ) {
        OutputStream output = null;
        try {
            String urlString = prefs.getString("rdtCheckUrl", mHttpURL);
            String guid = String.valueOf(java.util.UUID.randomUUID());
            String metaDataStr = "{\"UUID\":" + "\"" + guid + "\",\"Quality_parameters\":{\"brightness\":\"10\"},\"RDT_Type\":\"Flu_Audere\",\"Include_Proof\":\"True\"}";
            try {
                Httpok mr = new Httpok("img.jpg", imagebytes, urlString, metaDataStr,"label.json",labelByte);

                mr.delegate = this;
                mr.setCtx(getApplicationContext());
                mr.execute();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } finally {
            if (null != output) {
                try {
//                    Continue();
                    output.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    private Bitmap readFile(String filename) {
        String extStorageState =  mctx.getFilesDir().getAbsolutePath();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap = BitmapFactory.decodeFile(filename, options);
        Matrix matrix = new Matrix();

        matrix.postRotate(90);

        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 1280,720, true);

        Bitmap rotatedBitmap = Bitmap.createBitmap(scaledBitmap, 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight(), matrix, true);
        return rotatedBitmap;
    }
    @Override
    public void processFinish(String output) {
        MainActivity.removeServerUploadMessage();
        finish();

    }
}
