package com.fezrestia.android.helloworld;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;

import com.fezrestia.android.util.log.Log;

public class ThreadHandlerDebug extends BroadcastReceiver {
    public static final String TAG = "ThreadHandlerDebug";

    public static LocalThread mLocalThread;

    static {
        Log.logError(TAG, "static block : E");
        mLocalThread = new LocalThread();
        mLocalThread.start();
        Log.logError(TAG, "static block : X");
    }

    public ThreadHandlerDebug() {
        Log.logError(TAG, "CONSTRUCTOR()");
    }

    public static void check() {
        Log.logError(TAG, "check() : E");


        Log.logError(TAG, "check() : X");
    }

    @Override
    public void onReceive(Context context, Intent intent) {
        Log.logError(TAG, "onReceive()");

        Message msg = mLocalThread.mHandler.obtainMessage();
        msg.what = 1000;
        msg.obj = "HogeFugaPiyo";
        mLocalThread.mHandler.sendMessage(msg);
    }

}

class LocalThread extends Thread {
    public static final String TAG = "LocalThread";
    public Handler mHandler;

    /**
     * CONSTRUCTOR.
     */
    public LocalThread() {
        Log.logError(TAG, "CONSTRUCTOR : E");

        mHandler = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                Log.logError(TAG, "handleMessage() : " + msg.toString());
            }
        };

        Log.logError(TAG, "CONSTRUCTOR : X");
    }

    @Override
    public void run() {
        Log.logError(TAG, "run() : E");

        Looper.prepare();
        Looper.loop();

        Log.logError(TAG, "run() : X");
    }

}
