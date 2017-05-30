package com.fezrestia.android.helloworld;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

public class MainActivity extends Activity {
    public static final String TAG = "TraceLog";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.e(TAG, "onCreate() : E");
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        Log.e(TAG, "onCreate() : X");
    }

    @Override
    public void onResume() {
        Log.e(TAG, "onResume() : E");
        super.onResume();

//        ClockDebug.check();

        Log.e(TAG, "onResume() : X");
    }

    @Override
    public void onPause() {
        Log.e(TAG, "onPause() : E");

        // NOP.

        super.onPause();
        Log.e(TAG, "onPause() : X");
    }

    @Override
    public void onDestroy() {
        Log.e(TAG, "onDestroy() : E");

        // NOP.

        super.onDestroy();
        Log.e(TAG, "onDestroy() : X");
    }
}
