package com.fezrestia.android.helloworld.bluetooth;

import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.os.Handler;
import android.os.HandlerThread;

import com.fezrestia.android.util.log.Log;

import java.util.Deque;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Control BLE GATT Input/Output.
 */
public class BleGattIO {
    // Log tag.
    private static final String TAG = "BleGattIO";
    // Log flag.
    public static final boolean IS_DEBUG = false || Log.IS_DEBUG;

    private Deque<Task> mTaskDeque = null;

    private HandlerThread mHandlerThread = null;
    private Handler mHandler = null;

    private static final int RETRY_DELAY_MILLIS = 1000;

    private BleGattIOCallback mCallback = null;

    private final Object mCallbackLock = new Object();
    private final Object mIdleLock = new Object();

    // Status flag will be updated only once in release().
    private volatile boolean mIsActive = true;

    private ThreadSafeCounter mRequestCounter = null;

    // Request counter.
    private static class ThreadSafeCounter {
        private volatile int mCount;

        /**
         * Get current count.
         *
         * @return Current count.
         */
        public int get() {
            return mCount;
        }

        /**
         * Increment count.
         *
         * @return Incremented count.
         */
        public synchronized int increment() {
            return ++mCount;
        }
    }

    private enum IO_TARGET {
        CHARACTERISTIC,
        DESCRIPTOR,
    }

    private enum IO_TYPE {
        READ,
        WRITE,
    }

    /**
     * Callback interface for GATT I/O.
     */
    public interface BleGattIOCallback {
        void onGattCharaReadDone(BluetoothGattCharacteristic chara);
        void onGattCharaWriteDone(BluetoothGattCharacteristic chara);
        void onGattDescReadDone(BluetoothGattDescriptor desc);
        void onGattDescWriteDone(BluetoothGattDescriptor desc);
    }

    private static class Task {
        public final IO_TARGET ioTarget;
        public final IO_TYPE ioType;
        public final BluetoothGatt gatt;
        public final BluetoothGattCharacteristic chara;
        public final BluetoothGattDescriptor desc;

        /**
         * CONSTRUCTOR.
         */
        private Task(
                IO_TARGET ioTarget,
                IO_TYPE ioType,
                BluetoothGatt gatt,
                BluetoothGattCharacteristic chara,
                BluetoothGattDescriptor desc) {
            this.ioTarget = ioTarget;
            this.ioType = ioType;
            this.gatt = gatt;
            this.chara = chara;
            this.desc = desc;
        }

        @Override
        public String toString() {
            return "TASK = " + ioTarget + " / " + ioType + " / CHARA=" + chara + " / DESC=" +desc;
        }

        private static Task genCharaReadTask(
                BluetoothGatt gatt,
                BluetoothGattCharacteristic chara) {
            return new Task(IO_TARGET.CHARACTERISTIC, IO_TYPE.READ, gatt, chara, null);
        }

        private static Task genCharaWriteTask(
                BluetoothGatt gatt,
                BluetoothGattCharacteristic chara) {
            return new Task(IO_TARGET.CHARACTERISTIC, IO_TYPE.WRITE, gatt, chara, null);
        }

        private static Task genDescReadTask(
                BluetoothGatt gatt,
                BluetoothGattDescriptor desc) {
            return new Task(IO_TARGET.DESCRIPTOR, IO_TYPE.READ, gatt, null, desc);
        }

        private static Task genDescWriteTask(
                BluetoothGatt gatt,
                BluetoothGattDescriptor desc) {
            return new Task(IO_TARGET.DESCRIPTOR, IO_TYPE.WRITE, gatt, null, desc);
        }
    }

    /**
     * CONSTRUCTOR.
     */
    public BleGattIO() {
        if (IS_DEBUG) Log.logDebug(TAG, "CONSTRUCTOR : E");

        mTaskDeque = new ConcurrentLinkedDeque<>();
        mHandlerThread = new HandlerThread("BLE-IO");
        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper());

        // Start task queue polling.
        mHandler.post(new TaskProcessor());

        mRequestCounter = new ThreadSafeCounter();

        if (IS_DEBUG) Log.logDebug(TAG, "CONSTRUCTOR : X");
    }

    /**
     * Release all references.
     */
    public void release() {
        if (IS_DEBUG) Log.logDebug(TAG, "release() : E");

        mIsActive = false; // Updated only once.

        // Stop thread.
        synchronized(mCallbackLock) {
            mCallbackLock.notify();
        }
        synchronized(mIdleLock) {
            mIdleLock.notify();
        }
        mHandlerThread.quitSafely();
        try {
            mHandlerThread.join(RETRY_DELAY_MILLIS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        mTaskDeque.clear();
        mTaskDeque = null;
        mHandler = null;
        mHandlerThread = null;
        mRequestCounter = null;

        if (IS_DEBUG) Log.logDebug(TAG, "release() : X");
    }

    /**
     * Request to read characteristic.
     * Result will be called back async.
     *
     * @param gatt BluetoothGatt
     * @param chara BluetoothGattCharacteristic
     */
    public void requestReadChara(BluetoothGatt gatt, BluetoothGattCharacteristic chara) {
        if (IS_DEBUG) Log.logDebug(TAG, "requestReadChara() : CHARA=" + chara.getUuid());
        Task task = Task.genCharaReadTask(gatt, chara);
        request(task);
    }

    /**
     * Request to write characteristic.
     * Result will be called back async.
     *
     * @param gatt BluetoothGatt
     * @param chara BluetoothGattCharacteristic
     */
    public void requestWriteChara(BluetoothGatt gatt, BluetoothGattCharacteristic chara) {
        if (IS_DEBUG) Log.logDebug(TAG, "requestWriteChara() : CHARA=" + chara.getUuid());
        Task task = Task.genCharaWriteTask(gatt, chara);
        request(task);
    }

    /**
     * Request to read descriptor.
     *
     * @param gatt BluetoothGatt
     * @param desc BluetoothGattDescriptor
     */
    public void requestReadDesc(BluetoothGatt gatt, BluetoothGattDescriptor desc) {
        if (IS_DEBUG) Log.logDebug(TAG, "requestReadDesc() : DESC=" + desc.getUuid());
        Task task = Task.genDescReadTask(gatt, desc);
        request(task);
    }

    /**
     * Requst to write descriptor.
     *
     * @param gatt BluetoothGatt
     * @param desc BluetoothGattDescriptor
     */
    public void requestWriteDesc(BluetoothGatt gatt, BluetoothGattDescriptor desc) {
        if (IS_DEBUG) Log.logDebug(TAG, "requestWriteDesc() : DESC=" + desc.getUuid());
        Task task = Task.genDescWriteTask(gatt, desc);
        request(task);
    }

    private void request(Task task) {
        if (IS_DEBUG) Log.logDebug(TAG, "request() Notify to Idling : E");
        synchronized(mIdleLock) {
            mTaskDeque.addLast(task);
            mIdleLock.notify();
        }

        mRequestCounter.increment();

        if (IS_DEBUG) Log.logDebug(TAG, "request() Notify to Idling : X");
    }

    /**
     * Notify Read/Write request is completed to BleGattIO instance.
     */
    public void onRequestDone() {
        if (IS_DEBUG) Log.logDebug(TAG, "onRequestDone() Notify to Callback : E");
        synchronized(mCallbackLock) {
            mCallbackLock.notify();
        }
        if (IS_DEBUG) Log.logDebug(TAG, "onRequestDone() Notify to Callback : X");
    }

    private class TaskProcessor implements Runnable {
        private int mDoneCount = 0;

        @Override
        public void run() {
            if (IS_DEBUG) Log.logDebug(TAG, "TaskProcessor.run() : E");

            while (mIsActive) {
                // Check queued task.
                if (IS_DEBUG) Log.logDebug(TAG, "Idling wait for a while ...");
                synchronized (mIdleLock) {
                    if (mTaskDeque.isEmpty()) {
                        // Wait for a while.
                        try {
                            mIdleLock.wait(RETRY_DELAY_MILLIS);
                        } catch (InterruptedException e) {
                            // Check next loop.
                            continue;
                        }
                    }
                }
                if (IS_DEBUG) Log.logDebug(TAG, "Idling is DONE");

                // Get top task from deque.
                Task task = mTaskDeque.peekFirst();
                if (task == null) {
                    if (IS_DEBUG) Log.logDebug(TAG, "Peeked Task is null, keep idling.");
                    continue;
                }
                if (IS_DEBUG) Log.logDebug(TAG, "Do Task = " + task.toString());

                // Check readable/writable.
                boolean isOk = checkProperty(task);
                if (!isOk) {
                    if (IS_DEBUG) Log.logDebug(TAG, "Target is not readable/writable. Skip this.");

                    // Skip this task.
                    mTaskDeque.pollFirst();
                    continue;
                }

                if (IS_DEBUG) Log.logError(TAG, "Callback waiting ...");
                synchronized(mCallbackLock) {
                    // Do request I/O to GATT.
                    boolean isSuccess = doRequest(task); // Start waiting for callback if success.
                    // Check request success/fail.
                    if (isSuccess) {
                        try {
                            mCallbackLock.wait();

                            // Counted down. Result is successfully returned. Remove task from deque.
                            mTaskDeque.pollFirst();

                        } catch (InterruptedException e) {
                            if (IS_DEBUG) Log.logError(TAG, "Timeout to request read. ReTry.");
                            continue;
                        }
                    } else {
                        if (IS_DEBUG) Log.logError(TAG, "Failed to request read. ReTry.");
                        continue;
                    }
                }
                if (IS_DEBUG) Log.logError(TAG, "Callback waiting DONE.");

                // Task done successfully.
                ++mDoneCount;
                if (IS_DEBUG) Log.logError(TAG,
                        "Done/Request = " + mDoneCount + "/" + mRequestCounter.get());

                // Callback.
                if (mCallback != null) doCallback(task);
            } // while

            if (IS_DEBUG) Log.logDebug(TAG, "TaskProcessor.run() : X");
        }

        private boolean checkProperty(Task task) {
            boolean isOk = false;

            switch (task.ioTarget) {
                case CHARACTERISTIC:
                    switch (task.ioType) {
                        case READ:
                            isOk = (task.chara.getProperties()
                                    & BluetoothGattCharacteristic.PROPERTY_READ) != 0;
                            break;

                        case WRITE:
                            int flags = BluetoothGattCharacteristic.PROPERTY_WRITE
                                    | BluetoothGattCharacteristic.PROPERTY_WRITE_NO_RESPONSE;
                            isOk = (task.chara.getProperties() & flags) != 0;
                            break;

                        default:
                            throw new IllegalArgumentException("Unexpected IO_TYPE");
                    }
                    break;

                case DESCRIPTOR:
                    isOk = true;
                    break;

                default:
                    throw new IllegalArgumentException("Unexpected IO_TARGET");
            }

            return isOk;
        }

        private boolean doRequest(Task task) {
            boolean isSuccess = false;

            switch (task.ioTarget) {
                case CHARACTERISTIC:
                    switch (task.ioType) {
                        case READ:
                            isSuccess = task.gatt.readCharacteristic(task.chara);
                            break;

                        case WRITE:
                            isSuccess = task.gatt.writeCharacteristic(task.chara);
                            break;

                        default:
                            throw new IllegalArgumentException("Unexpected IO_TYPE");
                    }
                    break;

                case DESCRIPTOR:
                    switch (task.ioType) {
                        case READ:
                            isSuccess = task.gatt.readDescriptor(task.desc);
                            break;

                        case WRITE:
                            isSuccess = task.gatt.writeDescriptor(task.desc);
                            break;

                        default:
                            throw new IllegalArgumentException("Unexpected IO_TYPE");
                    }
                    break;

                default:
                    throw new IllegalArgumentException("Unexpected IO_TARGET");
            }

            return isSuccess;
        }

        private void doCallback(Task task) {
            switch (task.ioTarget) {
                case CHARACTERISTIC:
                    switch (task.ioType) {
                        case READ:
                            mCallback.onGattCharaReadDone(task.chara);
                            break;

                        case WRITE:
                            mCallback.onGattCharaWriteDone(task.chara);
                            break;

                        default:
                            throw new IllegalArgumentException("Unexpected IO_TYPE");
                    }
                    break;

                case DESCRIPTOR:
                    switch (task.ioType) {
                        case READ:
                            mCallback.onGattDescReadDone(task.desc);
                            break;

                        case WRITE:
                            mCallback.onGattDescWriteDone(task.desc);
                            break;

                        default:
                            throw new IllegalArgumentException("Unexpected IO_TYPE");
                    }
                    break;

                default:
                    throw new IllegalArgumentException("Unexpected IO_TARGET");
            }
        }
    }
}
