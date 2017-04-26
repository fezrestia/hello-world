package com.fezrestia.android.helloworld.bluetooth;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothClass;
import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;

import com.fezrestia.android.util.log.Log;

/**
 * Broadcast receiver for bluetooth related intent.
 */
public class BtIntentReceiver extends BroadcastReceiver {
    // Log Tag.
    private static final String TAG = "BtIntentReceiver";
    // Log flag.
    public static final boolean IS_DEBUG = false || Log.IS_DEBUG;

    @Override
    public void onReceive(Context context, Intent intent) {
        if (IS_DEBUG) Log.logDebug(TAG, "onReceive() : E");

        String action = intent.getAction();
        if (IS_DEBUG) Log.logDebug(TAG, "    ACTION = " + action);

        if (action == null) {
            if (IS_DEBUG) Log.logDebug(TAG, "Action is NULL");
            // NOP.
        } else switch (action) {
            case BluetoothAdapter.ACTION_STATE_CHANGED:
                String state = intent.getParcelableExtra(BluetoothAdapter.EXTRA_STATE);
                String prevState = intent.getParcelableExtra(BluetoothAdapter.EXTRA_STATE);

                if (IS_DEBUG) Log.logDebug(TAG, "    STATE = " + state);
                if (IS_DEBUG) Log.logDebug(TAG, "    Prev STATE = " + prevState);

                break;

            case BluetoothAdapter.ACTION_DISCOVERY_STARTED:
                // NOP.
                break;

            case BluetoothAdapter.ACTION_DISCOVERY_FINISHED:
                // NOP.
                break;

            case BluetoothDevice.ACTION_FOUND:
                BluetoothDevice btDevice = intent.getParcelableExtra(
                        BluetoothDevice.EXTRA_DEVICE);
                BluetoothClass btClass = intent.getParcelableExtra(
                        BluetoothDevice.EXTRA_CLASS);
                if (IS_DEBUG) BtBleLog.logBtDevice(TAG, btDevice);
                if (IS_DEBUG) BtBleLog.logBtClass(TAG, btClass);
                break;

            default:
                if (IS_DEBUG) Log.logDebug(TAG, "Unexpected Action = " + action);
                break;
        }

        if (IS_DEBUG) Log.logDebug(TAG, "onReceive() : X");
    }

    /**
     * Register this receiver to system.
     *
     * @param context Master context.
     */
    public void registerToSystem(Context context) {
        IntentFilter filter = new IntentFilter();
        filter.addAction(BluetoothAdapter.ACTION_STATE_CHANGED);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED);
        filter.addAction(BluetoothDevice.ACTION_FOUND);

        context.registerReceiver(this, filter);
    }

    /**
     * Unregister this receiver from system.
     *
     * @param context Master context.
     */
    public void unregisterBtReceiver(Context context) {
        context.unregisterReceiver(this);
    }
}
