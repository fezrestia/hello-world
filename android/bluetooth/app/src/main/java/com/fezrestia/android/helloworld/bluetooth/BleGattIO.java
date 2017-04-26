package com.fezrestia.android.helloworld.bluetooth;

import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.os.Handler;
import android.os.HandlerThread;

import com.fezrestia.android.util.log.Log;

import java.util.Deque;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

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

    private CountDownLatch mRequestDoneWait = null; // Max 1

    private enum IO_TARGET {
        CHARACTERISTIC,
        DESCRIPTOR,
    }

    private enum IO_TYPE {
        READ,
        WRITE,
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
        mTaskDeque = new ConcurrentLinkedDeque<>();
        mHandlerThread = new HandlerThread("BLE-IO");
        mHandlerThread.start();
        mHandler = new Handler(mHandlerThread.getLooper());
    }

    /**
     * Release all references.
     */
    public void release() {
        mTaskDeque.clear();
        mTaskDeque = null;

        if (mRequestDoneWait != null) {
            mRequestDoneWait.countDown();
            mRequestDoneWait = null;
        }

        mHandler = null;
        mHandlerThread.quitSafely();
        mHandlerThread = null;
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
        mTaskDeque.addLast(task);
        mHandler.post(new TaskProcessor());
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
        mTaskDeque.addLast(task);
        mHandler.post(new TaskProcessor());
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
        mTaskDeque.addLast(task);
        mHandler.post(new TaskProcessor());
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
        mTaskDeque.addLast(task);
        mHandler.post(new TaskProcessor());
    }

    /**
     * Notify Read/Write request is completed to BleGattIO instance.
     */
    public void onRequestDone() {
        if (IS_DEBUG) Log.logDebug(TAG, "onRequestDone()");
        if (mRequestDoneWait != null) {
            if (IS_DEBUG) Log.logDebug(TAG, "onRequestDone() : Do count down");
            mRequestDoneWait.countDown();
        }
    }

    private class TaskProcessor implements Runnable {
        @Override
        public void run() {
            if (IS_DEBUG) Log.logDebug(TAG, "TaskProcessor.run() : E");

            Task task = mTaskDeque.peekFirst();
            if (task == null) {
                // Task deque is already empty.
                if (IS_DEBUG) Log.logDebug(TAG, "TaskProcessor.run() : X : Already Empty");
                return;
            }

            mRequestDoneWait = new CountDownLatch(1);

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
            }

            if (isSuccess) {
                // Task is handled successfully. Remove it from deque.
                mTaskDeque.removeFirst();

                try {
                    mRequestDoneWait.await(5000, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    if (IS_DEBUG) Log.logError(TAG, "Timeouted to request read. ReTry.");

                    if (mHandler != null) mHandler.postDelayed(this, RETRY_DELAY_MILLIS);
                }
            } else {
                if (IS_DEBUG) Log.logError(TAG, "Failed to request read. ReTry.");

                mRequestDoneWait = null;

                if (mHandler != null) mHandler.postDelayed(this, RETRY_DELAY_MILLIS);
            }

            if (IS_DEBUG) Log.logDebug(TAG, "TaskProcessor.run() : X");
        }
    }
}
