package com.fezrestia.android.helloworld.bluetooth;

import android.bluetooth.BluetoothClass;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanRecord;
import android.bluetooth.le.ScanResult;
import android.os.ParcelUuid;

import com.fezrestia.android.util.ByteUtil;
import com.fezrestia.android.util.log.Log;

import java.util.List;
import java.util.Set;

/**
 * Log utils related for BT BLE.
 */
public class BtBleLog {

    /**
     * Log all BluetoothDevice.
     *
     * @param logtag Log tag string.
     * @param devices Set of BluetoothDevice
     */
    public static void logAllBluetoothDevices(String logtag, Set<BluetoothDevice> devices) {
        StringBuilder sb = new StringBuilder();

        if (devices.size() > 0) {
            for (BluetoothDevice device : devices) {
                sb.append("#### BluetoothDevice :\n");
                sb.append("         DEVICE = ").append(device.getName()).append("\n");
                sb.append("            MAC = ").append(device.getAddress()).append("\n");

                ParcelUuid[] uuids = device.getUuids();
                if (uuids.length > 0) {
                    for (ParcelUuid uuid : uuids) {
                        sb.append("           UUID = ").append(uuid.getUuid()).append("\n");
                    }
                } else {
                    sb.append("           UUID = NULL\n");
                }
            }
        } else {
            sb.append("#### There is NO BluetoothDevice.");
        }

        Log.logDebug(logtag, sb.toString());
    }

    /**
     * Log scan failed error code.
     *
     * @param logtag Log tag.
     * @param error Error code integer.
     */
    public static void logScanFailedErrorCode(String logtag, int error) {
        String errStr;
        switch (error) {
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
        Log.logDebug(logtag, errStr);
    }

    /**
     * Log BLE ScanResult.
     *
     * @param logtag Log tag.
     * @param result BLE ScanResult
     */
    public static void logScanResult(String logtag, ScanResult result) {
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

                String dataStr = ByteUtil.byte2Utf8(data);
                if (dataStr == null) dataStr = "N/A";
                sb.append("      DATA(str) = ").append(dataStr).append("\n");

                String hexStr = ByteUtil.byte2HexStr(data);
                if (hexStr == null) hexStr = "N/A";
                sb.append("      DATA(hex) = ").append(hexStr).append("\n");
            }
        } else {
            sb.append("   SERVICE UUID = NULL\n");
        }

        Log.logDebug(logtag, sb.toString());
    }

    /**
     * Log GATT services.
     *
     * @param gatt BluetoothGatt
     */
    public static void logAllGattServices(String logtag, BluetoothGatt gatt) {
        if (gatt.getServices() != null) {
            for (BluetoothGattService service : gatt.getServices()) {
                // Log parent service.
                Log.logDebug(logtag, "#### Parent Service:");
                logGattService(logtag, service);

                List<BluetoothGattService> included = service.getIncludedServices();
                if (included != null) {
                    for (BluetoothGattService s : included) {
                        Log.logDebug(logtag, "######## Included Service:");
                        logGattService(logtag, s);
                    }
                } else {
                    Log.logDebug(logtag, "######## inc.service = NULL");
                }
            }
        } else {
            Log.logDebug(logtag, "#### There is NO service.");
        }
    }

    /**
     * Log 1 GATT service.
     *
     * @param service BluetoothGattService
     */
    public static void logGattService(String logtag, BluetoothGattService service) {
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

                String valStr = ByteUtil.byte2HexStr(chara.getValue());
                if (valStr == null) valStr = "N/A";
                sb.append("          VALUE = ").append(valStr).append("\n");

                List<BluetoothGattDescriptor> descs = chara.getDescriptors();
                if (descs != null) {
                    for (BluetoothGattDescriptor desc : descs) {
                        sb.append("      DESC UUID = ").append(desc.getUuid()).append("\n");

                        byte[] data = desc.getValue();

                        String dataStr = ByteUtil.byte2Utf8(data);
                        if (dataStr == null) dataStr = "N/A";
                        sb.append("      DESC(str) = ").append(dataStr).append("\n");

                        String hexStr = ByteUtil.byte2HexStr(data);
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

        Log.logDebug(logtag, sb.toString());
    }

    /**
     * Log BluetoothGattCharacteristic.
     *
     * @param chara BluetoothGattCharacteristic
     */
    public static void logGattCharacteristic(String logtag, BluetoothGattCharacteristic chara) {
        StringBuilder sb = new StringBuilder();
        sb.append("BLE GATT Characteristic:\n");

        sb.append("           UUID = ").append(chara.getUuid()).append("\n");
        sb.append("     PERMISSION = ").append(getCharaPermLogStr(chara)).append("\n");
        sb.append("          PROPS = ").append(getCharaPropLogStr(chara)).append("\n");

        String valStr = ByteUtil.byte2HexStr(chara.getValue());
        if (valStr == null) valStr = "N/A";
        sb.append("          VALUE = ").append(valStr).append("\n");

        List<BluetoothGattDescriptor> descs = chara.getDescriptors();
        if (descs != null) {
            for (BluetoothGattDescriptor desc : descs) {
                byte[] data = desc.getValue();

                String dataStr = ByteUtil.byte2Utf8(data);
                if (dataStr == null) dataStr = "N/A";
                sb.append("      DESC(str) = ").append(dataStr).append("\n");

                String hexStr = ByteUtil.byte2HexStr(data);
                if (hexStr == null) hexStr = "N/A";
                sb.append("      DESC(hex) = ").append(hexStr).append("\n");
            }
        } else {
            sb.append("           DESC = NULL").append("\n");
        }

        Log.logDebug(logtag, sb.toString());
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

    /**
     * Log BluetoothDevice.
     *
     * @param logtag Tag string.
     * @param btDevice BluetoothDevice
     */
    public static void logBtDevice(String logtag, BluetoothDevice btDevice) {
        StringBuilder sb = new StringBuilder();

        sb.append("BluetoothDevice :\n");
        sb.append("        DEVICE = ").append(btDevice.getName()).append("\n");
        sb.append("           MAC = ").append(btDevice.getAddress()).append("\n");

        if (btDevice.getUuids() != null) {
            for (ParcelUuid uuid : btDevice.getUuids()) {
                sb.append("           UUID = ").append(uuid.getUuid()).append("\n");
            }
        } else {
            Log.logDebug(logtag, "           UUID = NULL\n");
        }

        Log.logDebug(logtag, sb.toString());
    }

    /**
     * Log BluetoothClass.
     *
     * @param logtag Tag string.
     * @param btClass BluetoothClass
     */
    public static void logBtClass(String logtag, BluetoothClass btClass) {
        StringBuilder sb = new StringBuilder();

        sb.append("BluetoothClass :\n");
        sb.append("    ").append(btClass.toString()).append("\n");

        Log.logDebug(logtag, sb.toString());
    }



}
