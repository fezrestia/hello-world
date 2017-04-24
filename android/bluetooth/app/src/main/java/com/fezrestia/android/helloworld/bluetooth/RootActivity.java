package com.fezrestia.android.helloworld.bluetooth;

import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothClass;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PermissionInfo;
import android.os.Bundle;
import android.os.ParcelUuid;
import android.view.View;
import android.widget.Button;

import com.fezrestia.android.util.log.Log;

import java.util.Set;

public class RootActivity extends Activity {
    // Log tag.
    public static final String TAG = "RootActivity";
    // Log flag.
    public static final boolean IS_DEBUG = true || Log.IS_DEBUG;

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
    protected void onStart() {
        if (IS_DEBUG) Log.logDebug(TAG, "onStart() : E");
        super.onStart();

        startBluetooth();

        if (IS_DEBUG) Log.logDebug(TAG, "onStart() : X");
    }

    @Override
    protected void onResume() {
        if (IS_DEBUG) Log.logDebug(TAG, "onResume() : E");
        super.onResume();

        registerBtReceiver();

        // Start device discovery.
        boolean isSuccess = mBtAdapter.startDiscovery();
        if (IS_DEBUG) Log.logDebug(TAG, "startDiscovery() state = " + isSuccess);

        if (IS_DEBUG) Log.logDebug(TAG, "onResume() : X");
    }

    @Override
    protected void onPause() {
        if (IS_DEBUG) Log.logDebug(TAG, "onPause() : E");

        // Stop device discovery.
        mBtAdapter.cancelDiscovery();

        unregisterBtReceiver();

        super.onPause();
        if (IS_DEBUG) Log.logDebug(TAG, "onPause() : X");
    }

    protected void onStop() {
        if (IS_DEBUG) Log.logDebug(TAG, "onStop() : E");

        stopBluetooth();

        super.onStop();
        if (IS_DEBUG) Log.logDebug(TAG, "onStop() : X");
    }

    @Override
    protected void onDestroy() {
        if (IS_DEBUG) Log.logDebug(TAG, "onDestroy() : E");

        // NOP.

        super.onDestroy();
        if (IS_DEBUG) Log.logDebug(TAG, "onDestroy() : X");
    }

    //////////// BLUETOOTH ///////////////////////////////////////////////////////////////////////

    private BluetoothManager mBtManager = null;

    private BluetoothAdapter mBtAdapter = null;

    private static final int REQUEST_TO_ENABLE_BLUETOOTH_ID = 100;

    private BroadcastReceiver mBtReceiver = null;

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_TO_ENABLE_BLUETOOTH_ID) {
            switch (resultCode) {
                case Activity.RESULT_OK:
                    // Enabled.

                    break;

                case Activity.RESULT_CANCELED:
                    // Disabled.

                    finish();
                    break;

                default:
                    break;
            }
        }
    }

    private void registerBtReceiver() {
        mBtReceiver = new BtIntentReceiver();
        IntentFilter filter = new IntentFilter();
        filter.addAction(BluetoothAdapter.ACTION_STATE_CHANGED);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_STARTED);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED);
        filter.addAction(BluetoothDevice.ACTION_FOUND);

        registerReceiver(mBtReceiver, filter);
    }

    private void unregisterBtReceiver() {
        if (mBtReceiver != null) {
            unregisterReceiver(mBtReceiver);
            mBtReceiver = null;
        }
    }

    private void startBluetooth() {
        if (IS_DEBUG) Log.logDebug(TAG, "startBluetooth() : E");

        mBtManager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        mBtAdapter = mBtManager.getAdapter();

        if (mBtAdapter == null) {
            // Not supported.
            if (IS_DEBUG) Log.logDebug(TAG, "startBluetooth() : Device does not support Bluetooth.");
        } else {
            // Supported.

            if (!mBtAdapter.isEnabled()) {
                // Disabled.

                // Enable Bluetooth.
                Intent enableIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
                startActivityForResult(enableIntent, REQUEST_TO_ENABLE_BLUETOOTH_ID);

            } else {
                // Enabled.

                // Check devices which already paired.
                Set<BluetoothDevice> pairedDevices = mBtAdapter.getBondedDevices();
                if (pairedDevices.size() > 0) {
                    if (IS_DEBUG) Log.logDebug(TAG, "#### Already paired devices");
                    for (BluetoothDevice device : pairedDevices) {
                        if (IS_DEBUG) Log.logDebug(TAG, "## DEVICE = " + device.getName());
                        if (IS_DEBUG) Log.logDebug(TAG, "      MAC = " + device.getAddress());
                        ParcelUuid[] uuids = device.getUuids();
                        if (uuids.length > 0) {
                            for (ParcelUuid uuid : uuids) {
                                if (IS_DEBUG) Log.logDebug(TAG, "     UUID = " + uuid.getUuid());
                            }
                        }
                    }
                }
            }
        }

        if (IS_DEBUG) Log.logDebug(TAG, "startBluetooth() : X");
    }

    private void stopBluetooth() {
        if (IS_DEBUG) Log.logDebug(TAG, "stopBluetooth() : E");

        // Stop device discovery.
        mBtAdapter = null;
        mBtManager = null;

        if (IS_DEBUG) Log.logDebug(TAG, "stopBluetooth() : X");
    }

    private class BtIntentReceiver extends BroadcastReceiver {
        private final String TAG = "BtIntentReceiver";

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

                    if (IS_DEBUG) Log.logDebug(TAG, "    DEVICE = " + btDevice.getName());
                    if (IS_DEBUG) Log.logDebug(TAG, "       MAC = " + btDevice.getAddress());
                    if (btDevice.getUuids() != null) {
                        for (ParcelUuid uuid : btDevice.getUuids()) {
                            if (IS_DEBUG) Log.logDebug(TAG, "      UUID = " + uuid.getUuid());
                        }
                    } else {
                        if (IS_DEBUG) Log.logDebug(TAG, "      UUID = NULL");
                    }

                    if (IS_DEBUG) Log.logDebug(TAG, "     CLASS = " + btClass.toString());

                    break;

                default:
                    if (IS_DEBUG) Log.logDebug(TAG, "Unexpected Action = " + action);
                    break;
            }

            if (IS_DEBUG) Log.logDebug(TAG, "onReceive() : X");
        }
    }





    private class Key1OnClickListener implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            if (IS_DEBUG) Log.logDebug(TAG, "Key1OnClickListener.onClick() : E");







            if (IS_DEBUG) Log.logDebug(TAG, "Key1OnClickListener.onClick() : X");
        }
    }
}
