package com.fezrestia.android.helloworld.bluetooth;

import android.app.Activity;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothClass;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanRecord;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.ParcelUuid;
import android.view.View;
import android.widget.Button;

import com.fezrestia.android.util.log.Log;

import org.xml.sax.SAXNotRecognizedException;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;

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

        Button keyLegacy = (Button) findViewById(R.id.legacy);
        keyLegacy.setOnClickListener(new KeyLegacyOnClickListener());
        Button keyBle = (Button) findViewById(R.id.ble);
        keyBle.setOnClickListener(new KeyBleOnClickListener());

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

        if (IS_DEBUG) Log.logDebug(TAG, "onResume() : X");
    }

    @Override
    protected void onPause() {
        if (IS_DEBUG) Log.logDebug(TAG, "onPause() : E");

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

    private BleDeviceScanner mBleDeviceScanner = null;

    private BleDeviceBinder mBleDeviceBinder = null;

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

        // BLE.
        mBleDeviceScanner = new BleDeviceScanner(
                mBtAdapter,
                new BleDeviceScannerCallbackImpl(),
                new Handler());

        if (IS_DEBUG) Log.logDebug(TAG, "startBluetooth() : X");
    }

    private void stopBluetooth() {
        if (IS_DEBUG) Log.logDebug(TAG, "stopBluetooth() : E");

        // Stop device discovery.
        mBtAdapter = null;
        mBtManager = null;

        // BLE.
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












    public static class BleDeviceScanner {
        private static final String TAG = "BleDeviceScanner";
        private BluetoothAdapter mBtAdapter = null;
        private boolean mIsScanning = false;
        private static final int SCAN_TIMEOUT_MILLIS = 10000;
        private Handler mWorkerHandler = null;
        private BleDeviceScannerCallback mCallback = null;

        /**
         * BleDeviceScanner done callback.
         */
        public interface BleDeviceScannerCallback {
            void onBleDeviceScanDone(BluetoothDevice bleDevice);
        }

        /**
         * CONSTRUCTOR.
         *
         * @param btAdapter
         * @param workerHandler
         */
        public BleDeviceScanner(
                BluetoothAdapter btAdapter,
                BleDeviceScannerCallback callback,
                Handler workerHandler) {
            mBtAdapter = btAdapter;
            mCallback = callback;
            mWorkerHandler = workerHandler;
        }

        /**
         * Release all references.
         */
        public void release() {
            mCallback = null;
            mBtAdapter = null;
            mWorkerHandler.removeCallbacks(mScanTimeoutTask);
            mWorkerHandler = null;
        }

        /**
         * Start scan.
         *
         * @return
         */
        public synchronized boolean start() {
            if (IS_DEBUG) Log.logDebug(TAG, "start() : E");

            if (mIsScanning) {
                // Already scanning.
                if (IS_DEBUG) Log.logDebug(TAG, "Already scanning");
                if (IS_DEBUG) Log.logDebug(TAG, "start() : X");
                return false;
            }

            mIsScanning = true;

            BluetoothLeScanner scanner = mBtAdapter.getBluetoothLeScanner();

            List<ScanFilter> filterList = new ArrayList<>();
            ScanFilter filterPowerUp = new ScanFilter.Builder()
//                    .setDeviceAddress("7C:EC:79:53:7A:3E")
//                    .setDeviceName("TailorToys PowerUp")
                    .build();
            filterList.add(filterPowerUp);
            ScanFilter filterPokeGoPlus = new ScanFilter.Builder()
                    .setDeviceName("Pokemon GO Plus")
                    .build();
            filterList.add(filterPokeGoPlus);

            ScanSettings settings = new ScanSettings.Builder()
                    .setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
                    .setMatchMode(ScanSettings.MATCH_MODE_AGGRESSIVE)
                    .setNumOfMatches(ScanSettings.MATCH_NUM_MAX_ADVERTISEMENT)
                    .setReportDelay(0)
                    .setScanMode(ScanSettings.SCAN_MODE_BALANCED)
                    .build();

            // Start.
            scanner.startScan(filterList, settings, mScanCallbackImpl);

            // Set timeout task.
            mWorkerHandler.postDelayed(mScanTimeoutTask, SCAN_TIMEOUT_MILLIS);

            if (IS_DEBUG) Log.logDebug(TAG, "start() : X");
            return true;
        }

        private final ScanCallbackImpl mScanCallbackImpl = new ScanCallbackImpl();
        private class ScanCallbackImpl extends ScanCallback {
            @Override
            public void onBatchScanResults(List<ScanResult> results) {
                super.onBatchScanResults(results);
                if (IS_DEBUG) Log.logDebug(TAG, "onBatchScanResults()");

                for (ScanResult result : results) {
                    handleResult(result);
                }
            }

            @Override
            public void onScanFailed(int errorCode) {
                super.onScanFailed(errorCode);
                if (IS_DEBUG) Log.logDebug(TAG, "onScanFailed()");

                String errStr;
                switch (errorCode) {
                    case ScanCallback.SCAN_FAILED_ALREADY_STARTED:
                        errStr = "ERR = SCAN_FAILED_ALREADY_STARTED";
                        break;
                    case ScanCallback.SCAN_FAILED_APPLICATION_REGISTRATION_FAILED:
                        errStr = "ERR = SCAN_FAILED_APPLICATION_REGISTRATION_FAILED";
                        break;
                    case ScanCallback.SCAN_FAILED_FEATURE_UNSUPPORTED:
                        errStr = "ERR = SCAN_FAILED_FEATURE_UNSUPPORTED";
                        break;
                    case ScanCallback.SCAN_FAILED_INTERNAL_ERROR:
                        errStr = "ERR = SCAN_FAILED_INTERNAL_ERROR";
                        break;
                    default:
                        errStr = "ERR = UNEXPECTED";
                        break;
                }
                if (IS_DEBUG) Log.logDebug(TAG, errStr);
            }

            @Override
            public void onScanResult(int callbackType, ScanResult result) {
                super.onScanResult(callbackType, result);
                if (IS_DEBUG) Log.logDebug(TAG, "onScanResult()");

                handleResult(result);
            }

            private void handleResult(ScanResult result) {
                if (IS_DEBUG) logScanResult(result);

                if (mCallback != null) {
                    mCallback.onBleDeviceScanDone(result.getDevice());
                }
            }
        }

        private final ScanTimeoutTask mScanTimeoutTask = new ScanTimeoutTask();
        private class ScanTimeoutTask implements Runnable {
            @Override
            public void run() {
                if (IS_DEBUG) Log.logDebug(TAG, "ScanTimeoutTask.run()");

                // Stop scan.
                stop();
            }
        }

        /**
         * Stop scan.
         */
        public synchronized void stop() {
            if (IS_DEBUG) Log.logDebug(TAG, "stop() : E");

            if (!mIsScanning) {
                if (IS_DEBUG) Log.logDebug(TAG, "Already stopped.");
                if (IS_DEBUG) Log.logDebug(TAG, "stop() : X");
                return;
            }

            mIsScanning = false;

            mBtAdapter.getBluetoothLeScanner().stopScan(mScanCallbackImpl);

            if (IS_DEBUG) Log.logDebug(TAG, "stop() : X");
        }

        /**
         * Now in scanning or not.
         *
         * @return
         */
        public synchronized boolean isScanning() {
            return mIsScanning;
        }

        private static void logScanResult(ScanResult result) {
            StringBuilder sb = new StringBuilder();
            sb.append("BLE Scan Result:\n");

            BluetoothDevice dev = result.getDevice();

            sb.append("DEVICE NAME/MAC = ") // [16 char] = [value]
                    .append(dev.getName())
                    .append(" / ")
                    .append(dev.getAddress())
                    .append("\n");

            if (dev.getUuids() != null) {
                for (ParcelUuid uuid : dev.getUuids()) {
                    sb.append("    DEVICE UUID = ").append(uuid).append("\n");
                }
            } else {
                sb.append("    DEVICE UUID = NULL").append("\n");
            }

            ScanRecord rec = result.getScanRecord();

            sb.append("    SCAN RECORD :\n");
            sb.append("    DEVICE NAME = ").append(rec.getDeviceName()).append("\n");
            if (rec.getServiceUuids() != null) {
                for (ParcelUuid uuid : rec.getServiceUuids()) {
                    sb.append("   SERVICE UUID = ").append(uuid.toString()).append("\n");

                    byte[] data = rec.getServiceData(uuid);

                    String dataStr = byte2Utf8(data);
                    if (dataStr == null) dataStr = "N/A";
                    sb.append("      DATA(str) = ").append(dataStr).append("\n");

                    String hexStr = byte2HexStr(data);
                    if (hexStr == null) hexStr = "N/A";
                    sb.append("      DATA(hex) = ").append(hexStr).append("\n");
                }
            } else {
                sb.append("   SERVICE UUID = NULL\n");
            }

            Log.logDebug(TAG, sb.toString());
        }
    }





    private class BleDeviceScannerCallbackImpl
            implements BleDeviceScanner.BleDeviceScannerCallback {
        @Override
        public void onBleDeviceScanDone(BluetoothDevice bleDevice) {
            if (IS_DEBUG) Log.logDebug(TAG, "onBleDeviceScanDone()");

            // Check device.
            if (bleDevice.getName().equals("TailorToys PowerUp")) {
                Log.logDebug(TAG, "#### Device Name Match ####");

                // Scan GATT.
                mBleDeviceBinder = new BleDeviceBinder(
                        RootActivity.this,
                        bleDevice,
                        new BleDeviceBinderCallbackImpl());
                mBleDeviceBinder.connect();
            }
        }
    }





    public static class BleDeviceBinder {
        private static final String TAG = "BleDeviceBinder";
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
                    if (IS_DEBUG) logDiscoveredServices(gatt);

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

        private static void logDiscoveredServices(BluetoothGatt gatt) {
            if (gatt.getServices() != null) {
                for (BluetoothGattService service : gatt.getServices()) {
                    // Log parent service.
                    if (IS_DEBUG) Log.logDebug(TAG, "#### Parent Service:");
                    if (IS_DEBUG) logBluetoothGattService(service);

                    List<BluetoothGattService> included = service.getIncludedServices();
                    if (included != null) {
                        for (BluetoothGattService s : included) {
                            if (IS_DEBUG) Log.logDebug(TAG, "######## Included Service:");
                            if (IS_DEBUG) logBluetoothGattService(s);
                        }
                    } else {
                        if (IS_DEBUG) Log.logDebug(TAG, "######## inc.service = NULL");
                    }
                }
            }
        }

        private static void logBluetoothGattService(BluetoothGattService service) {
            StringBuilder sb = new StringBuilder();
            sb.append("BLE GATT Service:\n");

            sb.append("   SERVICE UUID = ").append(service.getUuid()).append("\n");

            List<BluetoothGattCharacteristic> charas = service.getCharacteristics();
            if (charas != null) {
                sb.append("CHARACTERISTICS :\n");
                for (BluetoothGattCharacteristic chara : charas) {
                    sb.append("     CHARA UUID = ").append(chara.getUuid()).append("\n");
                    sb.append("     PERMISSION = ").append(getCharaPermLogStr(chara)).append("\n");
                    sb.append("          PROPS = ").append(getCharaPropLogStr(chara)).append("\n");

                    String valStr = byte2HexStr(chara.getValue());
                    if (valStr == null) valStr = "N/A";
                    sb.append("          VALUE = ").append(valStr).append("\n");

                    List<BluetoothGattDescriptor> descs = chara.getDescriptors();
                    if (descs != null) {
                        for (BluetoothGattDescriptor desc : descs) {
                            sb.append("      DESC UUID = ").append(desc.getUuid()).append("\n");

                            byte[] data = desc.getValue();

                            String dataStr = byte2Utf8(data);
                            if (dataStr == null) dataStr = "N/A";
                            sb.append("      DESC(str) = ").append(dataStr).append("\n");

                            String hexStr = byte2HexStr(data);
                            if (hexStr == null) hexStr = "N/A";
                            sb.append("      DESC(hex) = ").append(hexStr).append("\n");
                        }
                    } else {
                        sb.append("           DESC = NULL").append("\n");
                    }
                }
            } else {
                sb.append("CHARACTERISTICS : NULL\n");
            }

            Log.logDebug(TAG, sb.toString());
        }

    }



    private BluetoothGattService mTargetService;
    private BluetoothGattCharacteristic mTargetChara;

    private BluetoothGatt mBleGatt;

    private class BleDeviceBinderCallbackImpl implements BleDeviceBinder.BleDeviceBinderCallback {
        @Override
        public void onConnected(BluetoothGatt bleGatt) {
            if (IS_DEBUG) Log.logDebug(TAG, "onConnected()");


            mBleGatt = bleGatt;


            // PowerUp3.0 Service.
            mTargetService = bleGatt.getService(
                    UUID.fromString("86c3810e-f171-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### SERVICE = " + mTargetService.toString());

            // PowerUp3.0 Characteristics.
            mTargetChara = mTargetService.getCharacteristic(
                    UUID.fromString("86c3810e-0010-40d9-a117-26b300768cd6"));
            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + mTargetChara.toString());

//            BluetoothGattCharacteristic chara0020 = mTargetService.getCharacteristic(
//                    UUID.fromString("86c3810e-0020-40d9-a117-26b300768cd6"));
//            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara0020.toString());
//
//            BluetoothGattCharacteristic chara0021 = mTargetService.getCharacteristic(
//                    UUID.fromString("86c3810e-0021-40d9-a117-26b300768cd6"));
//            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara0021.toString());
//
//            BluetoothGattCharacteristic chara0040 = mTargetService.getCharacteristic(
//                    UUID.fromString("86c3810e-0040-40d9-a117-26b300768cd6"));
//            if (IS_DEBUG) Log.logDebug(TAG, "#### CHARA = " + chara0040.toString());

            // Request read.
            boolean ret = false;
            ret = bleGatt.readCharacteristic(mTargetChara);
            if (IS_DEBUG) Log.logDebug(TAG, "#### RET = " + ret);
//            ret = bleGatt.readCharacteristic(chara0020);
//            if (IS_DEBUG) Log.logDebug(TAG, "#### RET = " + ret);
//            ret = bleGatt.readCharacteristic(chara0021);
//            if (IS_DEBUG) Log.logDebug(TAG, "#### RET = " + ret);
//            ret = bleGatt.readCharacteristic(chara0040);
//            if (IS_DEBUG) Log.logDebug(TAG, "#### RET = " + ret);


        }

        @Override
        public void onDisconnected() {
            if (IS_DEBUG) Log.logDebug(TAG, "onDisconnected()");

            mBleGatt = null;

        }

        @Override
        public void onCharaRead(BluetoothGattCharacteristic chara) {
            if (IS_DEBUG) Log.logDebug(TAG, "onCharaRead()");
            if (IS_DEBUG) logBleGattChara(chara);




            mTargetChara.setValue(10, BluetoothGattCharacteristic.FORMAT_SINT8, 0);
            boolean ret = mBleGatt.writeCharacteristic(mTargetChara);
            if (IS_DEBUG) Log.logDebug(TAG, "onCharaWrite() #### ret = " + ret);



        }

        @Override
        public void onCharaWrite(BluetoothGattCharacteristic chara) {
            if (IS_DEBUG) Log.logDebug(TAG, "onCharaWrite()");
            if (IS_DEBUG) logBleGattChara(chara);




        }
    }




    private static void logBleGattChara(BluetoothGattCharacteristic chara) {
        StringBuilder sb = new StringBuilder();
        sb.append("BLE GATT Characteristic:\n");

        sb.append("           UUID = ").append(chara.getUuid()).append("\n");
        sb.append("     PERMISSION = ").append(getCharaPermLogStr(chara)).append("\n");
        sb.append("          PROPS = ").append(getCharaPropLogStr(chara)).append("\n");

        String valStr = byte2HexStr(chara.getValue());
        if (valStr == null) valStr = "N/A";
        sb.append("          VALUE = ").append(valStr).append("\n");

        List<BluetoothGattDescriptor> descs = chara.getDescriptors();
        if (descs != null) {
            for (BluetoothGattDescriptor desc : descs) {
                byte[] data = desc.getValue();

                String dataStr = byte2Utf8(data);
                if (dataStr == null) dataStr = "N/A";
                sb.append("      DESC(str) = ").append(dataStr).append("\n");

                String hexStr = byte2HexStr(data);
                if (hexStr == null) hexStr = "N/A";
                sb.append("      DESC(hex) = ").append(hexStr).append("\n");
            }
        } else {
            sb.append("           DESC = NULL").append("\n");
        }

        Log.logDebug(TAG, sb.toString());
    }


    private static String getCharaPermLogStr(BluetoothGattCharacteristic chara) {
        StringBuilder sb = new StringBuilder();

        int permission = chara.getPermissions();

        if (permission == 0) {
            sb.append("NO PERMISSION");
        } else {
            if ((BluetoothGattCharacteristic.PERMISSION_READ & permission) != 0) {
                sb.append("R/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_READ_ENCRYPTED & permission) != 0) {
                sb.append("REnc/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_READ_ENCRYPTED_MITM & permission) != 0) {
                sb.append("REncMitm/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_WRITE & permission) != 0) {
                sb.append("W/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_WRITE_ENCRYPTED & permission) != 0) {
                sb.append("WEnc/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_WRITE_ENCRYPTED_MITM & permission) != 0) {
                sb.append("WEncMitm/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_WRITE_SIGNED & permission) != 0) {
                sb.append("WSign/");
            }
            if ((BluetoothGattCharacteristic.PERMISSION_WRITE_SIGNED_MITM & permission) != 0) {
                sb.append("WSignMitm/");
            }
        }

        return sb.toString();
    }

    private static String getCharaPropLogStr(BluetoothGattCharacteristic chara) {
        StringBuilder sb = new StringBuilder();

        int props = chara.getProperties();

        if (props == 0) {
            sb.append("NO PROPERTY");
        } else {
            if ((BluetoothGattCharacteristic.PROPERTY_BROADCAST & props) != 0) {
                sb.append("BROADCAST/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_EXTENDED_PROPS & props) != 0) {
                sb.append("EXTENDED_PROPS/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_INDICATE & props) != 0) {
                sb.append("INDICATE/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_NOTIFY & props) != 0) {
                sb.append("NOTIFY/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_READ & props) != 0) {
                sb.append("READ/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_SIGNED_WRITE & props) != 0) {
                sb.append("SIGNED_WRITE/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_WRITE & props) != 0) {
                sb.append("WRITE/");
            }
            if ((BluetoothGattCharacteristic.PROPERTY_WRITE_NO_RESPONSE & props) != 0) {
                sb.append("WRITE_NO_RESPONSE/");
            }
        }

        return sb.toString();
    }




    private static String byte2Utf8(byte[] data) {
        if (data == null) return null;

        try {
            return new String(data, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            return null;
        }
    }

    private static String byte2HexStr(byte[] data) {
        if (data == null) return null;

        String hexStr = "";
        for (byte b : data) {
            hexStr += Integer.toHexString(b & 0xFF);
        }
        return hexStr;
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



    private class KeyBleOnClickListener implements View.OnClickListener {
        @Override
        public void onClick(View view) {
            if (IS_DEBUG) Log.logDebug(TAG, "KeyBleOnClickListener.onClick() : E");

            if (mBleDeviceScanner.isScanning()) {
                if (IS_DEBUG) Log.logDebug(TAG, "Stop BLE scan.");
                mBleDeviceScanner.stop();
            } else {
                if (IS_DEBUG) Log.logDebug(TAG, "Start BLE scan.");
                mBleDeviceScanner.start();
            }

            if (IS_DEBUG) Log.logDebug(TAG, "KeyBleOnClickListener.onClick() : X");
        }
    }
}
