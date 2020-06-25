package com.iprd.rdtcamera;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import static android.os.SystemClock.sleep;

public class LabelScreen extends AppCompatActivity {
    private ImageView clickedPicture;
    private Button submit;
    private Button back;
    private RadioButton positive;
    private RadioButton invalid;
    private RadioButton negative;
    private Bitmap recievedImage;
    private Context mctx;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mctx = getApplicationContext();
        setContentView(R.layout.activity_label_screen);
        clickedPicture = (ImageView) findViewById(R.id.rdtImage);
        back = (Button) findViewById(R.id.back);
        submit = (Button) findViewById(R.id.submit);

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

                Intent nextIntent = new Intent(mctx, MainActivity.class);
                nextIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);

                startActivity(nextIntent);
                finish();
            }
        });
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
}
