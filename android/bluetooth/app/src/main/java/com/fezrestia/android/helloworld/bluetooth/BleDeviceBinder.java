package com.fezrestia.android.helloworld.bluetooth;

import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothProfile;
import android.content.Context;

import com.fezrestia.android.util.log.Log;

/**
 * BLE device binder.
 */
public class BleDeviceBinder {
    // Log tag.
    private static final String TAG = "BleDeviceBinder";
    // Log flag.
    public static final boolean IS_DEBUG = false || Log.IS_DEBUG;

    private Context mContext = null;
    private BluetoothDevice mBleDevice = null;
    private BluetoothGatt mBleGatt = null;
    private BleDeviceBinderCallback mCallback = null;

    /**
     * BleDeviceBinder callback.
     */
    public interface BleDeviceBinderCallback {
        void onConnected(BluetoothGatt bleGatt);
        void onDisconnected();
        void onCharaRead(BluetoothGattCharacteristic chara);
        void onCharaWrite(BluetoothGattCharacteristic chara);



    }

    /**
     * CONSTRUCTOR.
     *
     * @param context
     * @param bleDevice
     * @param callback
     */
    public BleDeviceBinder(
            Context context,
            BluetoothDevice bleDevice,
            BleDeviceBinderCallback callback) {
        mContext = context;
        mBleDevice = bleDevice;
        mCallback = callback;
    }

    /**
     * Disconnect deivce and release all references.
     */
    public void release() {
        mCallback = null;
        if (mBleGatt != null) {
            mBleGatt.close();
            mBleGatt = null;
        }
        mBleDevice = null;
    }

    public void connect() {
        if (IS_DEBUG) Log.logDebug(TAG, "connect() : E");

        mBleGatt = mBleDevice.connectGatt(
                mContext,
                true,
                new BleGattCallback(),
                BluetoothDevice.TRANSPORT_AUTO);

        if (IS_DEBUG) Log.logDebug(TAG, "connect() : X");
    }

    private class BleGattCallback extends BluetoothGattCallback {
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            super.onConnectionStateChange(gatt, status, newState);
            if (IS_DEBUG) Log.logDebug(TAG, "BleGattCallback.onConnectionStateChange()");

            switch (newState) {
                case BluetoothProfile.STATE_CONNECTED:
                    if (IS_DEBUG) Log.logDebug(TAG, "STATE_CONNECTED");

                    // Start service discovery.
                    gatt.discoverServices();
                    break;

                case BluetoothProfile.STATE_DISCONNECTED:
                    if (IS_DEBUG) Log.logDebug(TAG, "STATE_DISCONNECTED");

                    // Callback.
                    if (mCallback != null) mCallback.onDisconnected();
                    break;
            }
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            super.onServicesDiscovered(gatt, status);
            if (IS_DEBUG) Log.logDebug(TAG, "BleGattCallback.onServicesDiscovered()");

            if (status == BluetoothGatt.GATT_SUCCESS) {
                if (IS_DEBUG) Log.logDebug(TAG, "stats = SUCCESS");
                if (IS_DEBUG) BtBleLog.logAllGattServices(TAG, gatt);

                // Callback.
                if (mCallback != null) mCallback.onConnected(gatt);
            }
        }

        @Override
        public void onCharacteristicRead(
                BluetoothGatt gatt,
                BluetoothGattCharacteristic characteristic,
                int status) {
            super.onCharacteristicRead(gatt, characteristic, status);
            if (IS_DEBUG) Log.logDebug(TAG, "BleGattCallback.onCharacteristicRead()");

            if (status == BluetoothGatt.GATT_SUCCESS) {
                if (IS_DEBUG) Log.logDebug(TAG, "    status == GATT_SUCCESS");
            } else {
                if (IS_DEBUG) Log.logDebug(TAG, "    status != GATT_SUCCESS");
            }

            // Callback.
            if (mCallback != null) mCallback.onCharaRead(characteristic);
        }

        @Override
        public void onCharacteristicWrite(
                BluetoothGatt gatt,
                BluetoothGattCharacteristic characteristic,
                int status) {
            super.onCharacteristicWrite(gatt, characteristic, status);
            if (IS_DEBUG) Log.logDebug(TAG, "BleGattCallback.onCharacteristicWrite()");

            if (status == BluetoothGatt.GATT_SUCCESS) {
                if (IS_DEBUG) Log.logDebug(TAG, "    status == GATT_SUCCESS");
            } else {
                if (IS_DEBUG) Log.logDebug(TAG, "    status != GATT_SUCCESS");
            }

            // Callback.
            if (mCallback != null) mCallback.onCharaWrite(characteristic);
        }



    }
}
