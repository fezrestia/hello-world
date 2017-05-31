package com.fezrestia.android.helloworld;

import android.app.Activity;
import android.os.Bundle;

import com.fezrestia.android.util.log.Log;

public class MainActivity extends Activity {
    public static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.logError(TAG, "onCreate() : E");
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        Log.logError(TAG, "onCreate() : X");
    }

    @Override
    public void onResume() {
        Log.logError(TAG, "onResume() : E");
        super.onResume();

//        ClockDebug.check();

        Log.logError(TAG, "onResume() : X");
    }

    @Override
    public void onPause() {
        Log.logError(TAG, "onPause() : E");

        // NOP.

        super.onPause();
        Log.logError(TAG, "onPause() : X");
    }

    @Override
    public void onDestroy() {
        Log.logError(TAG, "onDestroy() : E");

        // NOP.

        super.onDestroy();
        Log.logError(TAG, "onDestroy() : X");
    }
}
