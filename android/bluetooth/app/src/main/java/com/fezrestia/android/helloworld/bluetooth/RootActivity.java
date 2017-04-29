package com.fezrestia.android.helloworld.bluetooth;

import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.SeekBar;

import com.fezrestia.android.util.log.Log;

import java.util.List;
import java.util.ArrayList;
import java.util.UUID;

public class RootActivity extends Activity {
    // Log tag.
    public static final String TAG = "RootActivity";
    // Log flag.
    public static final boolean IS_DEBUG = false || Log.IS_DEBUG;

    private Handler mHandler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        if (IS_DEBUG) Log.logDebug(TAG, "onCreate() : E");
        super.onCreate(savedInstanceState);

        setContentView(R.layout.root_layout);

        Button keyLegacy = (Button) findViewById(R.id.legacy);
        keyLegacy.setOnClickListener(new KeyLegacyOnClickListener());
        Button keyBleSetUp = (Button) findViewById(R.id.ble_setup);
        keyBleSetUp.setOnClickListener(new KeyBleSetUpOnClickListener());
        Button keyBleDrive0 = (Button) findViewById(R.id.ble_drive_0);
        keyBleDrive0.setOnClickListener(new KeyBleDrive0OnClickListener());

        SeekBar sliderEngine = (SeekBar) findViewById(R.id.engine);
        sliderEngine.setOnSeekBarChangeListener(new EngineSliderListener());

        SeekBar sliderRudder = (SeekBar) findViewById(R.id.rudder);
        sliderRudder.setOnSeekBarChangeListener(new RudderSliderListener());

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

        mBtReceiver = new BtIntentReceiver();
        mBtReceiver.registerToSystem(this);

        if (IS_DEBUG) Log.logDebug(TAG, "onResume() : X");
    }

    @Override
    protected void onPause() {
        if (IS_DEBUG) Log.logDebug(TAG, "onPause() : E");

        mBtReceiver.unregisterBtReceiver(this);
        mBtReceiver = null;

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

    private static final int REQUEST_TO_ENABLE_BLUETOOTH_ID = 100;

    // Android Framework.
    private BluetoothManager mBtManager = null;
    private BluetoothAdapter mBtAdapter = null;

    // Extension.
    private BtIntentReceiver mBtReceiver = null;
    private BleDeviceScanner mBleDeviceScanner = null;
    private BleDeviceBinder mBleDeviceBinder = null;
    private BleGattIO mBleGattIO = null;

    // Target.
    private BluetoothDevice mTargetDevice;
    private BluetoothGatt mTargetGatt;
    private BluetoothGattService mTargetService;
    private List<BluetoothGattCharacteristic> mTargetCharaList = new ArrayList<>();
    private List<BluetoothGattDescriptor> mTargetDescList = new ArrayList<>();

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

    private void startBluetooth() {
        if (IS_DEBUG) Log.logDebug(TAG, "startBluetooth() : E");

        mBtManager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        mBtAdapter = mBtManager.getAdapter();

        if (mBtAdapter == null) {
            // Not supported.
            if (IS_DEBUG) Log.logDebug(TAG, "Device does not support Bluetooth.");
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
                BtBleLog.logAllBluetoothDevices(TAG, mBtAdapter.getBondedDevices());
            }
        }

        // BLE.
        mBleDeviceScanner = new BleDeviceScanner(
                mBtAdapter,
                new BleDeviceScannerCallbackImpl(),
                new Handler());
        mBleGattIO = new BleGattIO();

        if (IS_DEBUG) Log.logDebug(TAG, "startBluetooth() : X");
    }

    private void stopBluetooth() {
        if (IS_DEBUG) Log.logDebug(TAG, "stopBluetooth() : E");

        // Stop device discovery.
        mBtAdapter = null;
        mBtManager = null;

        // BLE.
        if (mBleGattIO != null) {
            mBleGattIO.release();
            mBleGattIO = null;
        }
        if (mBleDeviceBinder != null) {
            mBleDeviceBinder.release();
            mBleDeviceBinder = null;
        }
        if (mBleDeviceScanner != null) {
            mBleDeviceScanner.release();
            mBleDeviceScanner = null;
        }

        if (IS_DEBUG) Log.logDebug(TAG, "stopBluetooth() : X");
    }

    private class BleDeviceScannerCallbackImpl
            implements BleDeviceScanner.BleDeviceScannerCallback {
        @Override
        public void onBleDeviceScanDone(BluetoothDevice bleDevice) {
            if (IS_DEBUG) Log.logDebug(TAG, "onBleDeviceScanDone()");

            // Check device.
            if (bleDevice.getName() != null && bleDevice.getName().equals("TailorToys PowerUp")) {
                Log.logDebug(TAG, "######## Device Name Match ########");

                mTargetDevice = bleDevice;

                // Connect to BLE.
                mBleDeviceBinder = new BleDeviceBinder(
                        RootActivity.this,
                        mTargetDevice,
                        new BleDeviceBinderCallbackImpl());
                mBleDeviceBinder.connect();
            }
        }
    }

    private class BleDeviceBinderCallbackImpl implements BleDeviceBinder.BleDeviceBinderCallback {
        @Override
        public void onConnected(BluetoothGatt bleGatt) {
            if (IS_DEBUG) Log.logDebug(TAG, "onConnected()");

            mTargetGatt = bleGatt;

            List<BluetoothGattService> services = mTargetGatt.getServices();
            List<BluetoothGattCharacteristic> charas = new ArrayList<>();
            List<BluetoothGattDescriptor> descs = new ArrayList<>();

            // All elements.
            for (BluetoothGattService service : services) {
                if (service.getCharacteristics() != null) {
                    charas.addAll(service.getCharacteristics());
                }
            }
            for (BluetoothGattCharacteristic chara : charas) {
                if (chara.getDescriptors() != null) {
                    descs.addAll(chara.getDescriptors());
                }
            }
            // Request read.
            for (BluetoothGattCharacteristic chara : charas) {
                mBleGattIO.requestReadChara(mTargetGatt, chara);
            }
            for (BluetoothGattDescriptor desc : descs) {
                mBleGattIO.requestReadDesc(mTargetGatt, desc);
            }

            // PowerUp3.0 Service.
            mTargetService = bleGatt.getService(
                    UUID.fromString("86c3810e-f171-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### SERVICE = " + mTargetService.toString());

            // PowerUp3.0 Characteristics.
            BluetoothGattCharacteristic chara;

            chara = mTargetService.getCharacteristic(
                    UUID.fromString("86c3810e-0010-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara.toString());
            mTargetCharaList.add(chara);

            chara = mTargetService.getCharacteristic(
                    UUID.fromString("86c3810e-0020-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara.toString());
            mTargetCharaList.add(chara);

            chara = mTargetService.getCharacteristic(
                    UUID.fromString("86c3810e-0021-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara.toString());
            mTargetCharaList.add(chara);

            chara = mTargetService.getCharacteristic(
                    UUID.fromString("86c3810e-0040-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara.toString());
            mTargetCharaList.add(chara);
        }

        @Override
        public void onDisconnected() {
            if (IS_DEBUG) Log.logDebug(TAG, "onDisconnected()");

            // NOP.
        }

        @Override
        public void onCharaRead(boolean isSuccess, BluetoothGattCharacteristic chara) {
            if (IS_DEBUG) Log.logDebug(TAG, "onCharaRead()");
//            if (IS_DEBUG) BtBleLog.logAllGattServices(TAG, mTargetGatt);
//            if (IS_DEBUG) BtBleLog.logGattService(TAG, mTargetService);
            if (IS_DEBUG) BtBleLog.logGattCharacteristic(TAG, chara);

            mBleGattIO.onRequestDone();
        }

        @Override
        public void onCharaWrite(boolean isSuccess, BluetoothGattCharacteristic chara) {
            if (IS_DEBUG) Log.logDebug(TAG, "onCharaWrite()");
//            if (IS_DEBUG) BtBleLog.logAllGattServices(TAG, mTargetGatt);
//            if (IS_DEBUG) BtBleLog.logGattService(TAG, mTargetService);
            if (IS_DEBUG) BtBleLog.logGattCharacteristic(TAG, chara);

            mBleGattIO.onRequestDone();
        }

        @Override
        public void onDescRead(boolean isSuccess, BluetoothGattDescriptor desc) {
            if (IS_DEBUG) Log.logDebug(TAG, "onDescRead()");
//            if (IS_DEBUG) BtBleLog.logAllGattServices(TAG, mTargetGatt);
//            if (IS_DEBUG) BtBleLog.logGattService(TAG, mTargetService);
            if (IS_DEBUG) BtBleLog.logGattDescriptor(TAG, desc);

            mBleGattIO.onRequestDone();
        }

        @Override
        public void onDescWrite(boolean isSuccess, BluetoothGattDescriptor desc) {
            if (IS_DEBUG) Log.logDebug(TAG, "onDescWrite()");
//            if (IS_DEBUG) BtBleLog.logAllGattServices(TAG, mTargetGatt);
//            if (IS_DEBUG) BtBleLog.logGattService(TAG, mTargetService);
            if (IS_DEBUG) BtBleLog.logGattDescriptor(TAG, desc);

            mBleGattIO.onRequestDone();
        }



    }



    //// UI EVENT HANDLER ////////////////////////////////////////////////////////////////////////

    private class KeyLegacyOnClickListener implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            if (IS_DEBUG) Log.logDebug(TAG, "KeyLegacyOnClickListener.onClick() : E");

            if (mBtAdapter.isDiscovering()) {
                // Stop device discovery.
                mBtAdapter.cancelDiscovery();
                if (IS_DEBUG) Log.logDebug(TAG, "cancelDiscovery()");
            } else {
                // Start device discovery.
                boolean isSuccess = mBtAdapter.startDiscovery();
                if (IS_DEBUG) Log.logDebug(TAG, "startDiscovery() state = " + isSuccess);
            }

            if (IS_DEBUG) Log.logDebug(TAG, "KeyLegacyOnClickListener.onClick() : X");
        }
    }

    private class KeyBleSetUpOnClickListener implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            if (IS_DEBUG) Log.logDebug(TAG, "KeyBleSetUpOnClickListener.onClick() : E");

            if (mBleDeviceScanner.isScanning()) {
                if (IS_DEBUG) Log.logDebug(TAG, "Stop BLE scan.");
                mBleDeviceScanner.stop();
            } else {
                if (IS_DEBUG) Log.logDebug(TAG, "Start BLE scan.");
                mBleDeviceScanner.start();
            }

            if (IS_DEBUG) Log.logDebug(TAG, "KeyBleSetUpOnClickListener.onClick() : X");
        }
    }

    private class KeyBleDrive0OnClickListener implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            if (IS_DEBUG) Log.logDebug(TAG, "KeyBleDrive0OnClickListener.onClick() : E");

            mHandler.post(new EngineUpdateTask());

            if (IS_DEBUG) Log.logDebug(TAG, "KeyBleDrive0OnClickListener.onClick() : X");
        }
    }

    int mValue = 0;
    int mStep = 10;

    private class EngineUpdateTask implements Runnable {
        @Override
        public void run() {

            if (mTargetCharaList.isEmpty()) {
                return;
            }

            BluetoothGattCharacteristic chara = mTargetCharaList.get(0);

            mValue += mStep;
            if (mValue >= 250) {
                mStep = -10;
            }
            if (mValue < 0) {
                mValue = 0;
                mStep = 10;
            }

            chara.setValue(mValue, BluetoothGattCharacteristic.FORMAT_SINT8, 0);
            mBleGattIO.requestWriteChara(mTargetGatt, chara);

            if (mValue != 0) {
                mHandler.postDelayed(this, 1000);
            }
        }
    }

    private class EngineSliderListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
            if (IS_DEBUG) Log.logDebug(TAG,
                    "EngineSliderListener.onProgressChanged() : VAL = " + progress);

            if (mTargetCharaList == null || mTargetCharaList.isEmpty()) {
                return;
            }

            BluetoothGattCharacteristic chara = mTargetCharaList.get(0);
            chara.setValue(progress, BluetoothGattCharacteristic.FORMAT_SINT8, 0);
            mBleGattIO.requestWriteChara(mTargetGatt, chara);
        }

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {
            // Touch down on slider.
        }

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {
            // Touch up from slider.
        }
    }

    private class RudderSliderListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
            if (IS_DEBUG) Log.logDebug(TAG,
                    "RudderSliderListener.onProgressChanged() : VAL = " + progress);

            if (mTargetCharaList == null || mTargetCharaList.isEmpty()) {
                return;
            }

            progress = progress - 128; // 0 = center

            BluetoothGattCharacteristic chara = mTargetCharaList.get(2);
            chara.setValue(progress, BluetoothGattCharacteristic.FORMAT_SINT8, 0);
            mBleGattIO.requestWriteChara(mTargetGatt, chara);
        }

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {
            // Touch down on slider.
        }

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {
            // Touch up from slider.
        }
    }



}
