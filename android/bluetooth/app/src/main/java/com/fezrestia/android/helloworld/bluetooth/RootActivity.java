package com.fezrestia.android.helloworld.bluetooth;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.fezrestia.android.util.log.Log;

public class RootActivity extends Activity {
    // Log tag.
    public static final String TAG = "RootActivity";
    // Log flag.
    public static final boolean IS_DEBUG = false || Log.IS_DEBUG;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (IS_DEBUG) Log.logDebug(TAG, "onCreate() : E");
        super.onCreate(savedInstanceState);

        setContentView(R.layout.root_layout);

        Button key1 = (Button) findViewById(R.id.key1);
        key1.setOnClickListener(new Key1OnClickListener());

        if (IS_DEBUG) Log.logDebug(TAG, "onCreate() : X");
    }

    @Override
    protected void onResume() {
        if (IS_DEBUG) Log.logDebug(TAG, "onResume() : E");
        super.onResume();

        // NOP.

        if (IS_DEBUG) Log.logDebug(TAG, "onResume() : X");
    }

    @Override
    protected void onPause() {
        if (IS_DEBUG) Log.logDebug(TAG, "onPause() : E");

        // NOP.

        super.onPause();
        if (IS_DEBUG) Log.logDebug(TAG, "onPause() : X");
    }

    @Override
    protected void onDestroy() {
        if (IS_DEBUG) Log.logDebug(TAG, "onDestroy() : E");

        // NOP.

        super.onDestroy();
        if (IS_DEBUG) Log.logDebug(TAG, "onDestroy() : X");
    }

    private class Key1OnClickListener implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            Log.logDebug(TAG, "Key1OnClickListener.onClick() : E");







            Log.logDebug(TAG, "Key1OnClickListener.onClick() : X");
        }
    }
}
