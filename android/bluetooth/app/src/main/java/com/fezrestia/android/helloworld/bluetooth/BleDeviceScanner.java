package com.fezrestia.android.helloworld.bluetooth;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.os.Handler;

import com.fezrestia.android.util.log.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * Scan BLE devices.
 */
public class BleDeviceScanner {
    // Log tag.
    public static final String TAG = "BleDeviceScanner";
    // Log flag.
    public static final boolean IS_DEBUG = false || Log.IS_DEBUG;

    private BluetoothAdapter mBtAdapter = null;

    private boolean mIsScanning = false;

    private static final int SCAN_TIMEOUT_MILLIS = 12000; // 12sec

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
        ScanFilter filter = new ScanFilter.Builder()
//                    .setDeviceAddress("7C:EC:79:53:7A:3E") // Sample
//                    .setDeviceName("TailorToys PowerUp") // Sample
                .build();
        filterList.add(filter);

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
            if (IS_DEBUG) BtBleLog.logScanFailedErrorCode(TAG, errorCode);

            // NOP.
        }

        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            super.onScanResult(callbackType, result);
            if (IS_DEBUG) Log.logDebug(TAG, "onScanResult()");

            handleResult(result);
        }

        private void handleResult(ScanResult result) {
            if (IS_DEBUG) BtBleLog.logScanResult(TAG, result);

            if (mCallback != null) mCallback.onBleDeviceScanDone(result.getDevice());
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
}
