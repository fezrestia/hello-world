package com.fezrestia.android.helloworld;

import android.util.Log;

import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.TimeZone;

public class ClockDebug {
    public static final String TAG = "ClockDebug";

    public static void check() {
        Log.e(TAG, "check() : E");

        // Current UTC.
        Calendar curUtc = new GregorianCalendar(TimeZone.getTimeZone("GMT"));
        Date curUtcDate = curUtc.getTime();
//        Log.e(TAG, "   curUtcDate : year  = " + (curUtcDate.getYear() + 1900));
//        Log.e(TAG, "   curUtcDate : month = " + curUtcDate.getMonth());
//        Log.e(TAG, "   curUtcDate : day   = " + curUtcDate.getDate());
//        Log.e(TAG, "   curUtcDate : hour  = " + curUtcDate.getHours());
//        Log.e(TAG, "   curUtcDate : min   = " + curUtcDate.getMinutes());
//        Log.e(TAG, "   curUtcDate : sec   = " + curUtcDate.getSeconds());
//        curUtc.clear();
        curUtc.set(
                curUtcDate.getYear() + 1900,
                curUtcDate.getMonth(),
                curUtcDate.getDate(),
                curUtcDate.getHours(),
                curUtcDate.getMinutes(),
                curUtcDate.getSeconds());
        Log.e(TAG, "curUtc      = " + getCalLogStr(curUtc));
        Log.e(TAG, "curUtc.getTimeInMillis()         = " + curUtc.getTimeInMillis());
        Log.e(TAG, "curUtc.getTime().getTime()       = " + curUtc.getTime().getTime());




/*
        Log.e(TAG, "--------------------------------------------------------");
        // UTC 0
        Calendar utcZero = Calendar.getInstance(TimeZone.getTimeZone("GMT"));
        utcZero.set(1970, 0, 1, 0, 0, 0);
        utcZero.set(Calendar.MILLISECOND, 0);
        Log.e(TAG, "utcZero   = " + getCalLogStr(utcZero));
        Log.e(TAG, "utcZero.getTimeInMillis()   = " + utcZero.getTimeInMillis());
        Log.e(TAG, "getUtcMillisFrom(utcZero)   = " + getUtcMillisFrom(utcZero));

        // UTC Local
        Calendar utcLocal = Calendar.getInstance(TimeZone.getDefault());
        utcLocal.set(1970, 0, 1, 0, 0, 0);
        utcLocal.set(Calendar.MILLISECOND, 0);
        Log.e(TAG, "utcLocal  = " + getCalLogStr(utcLocal));
        Log.e(TAG, "utcLocal.getTimeInMillis()  = " + utcLocal.getTimeInMillis());
        Log.e(TAG, "getUtcMillisFrom(utcLocal)  = " + getUtcMillisFrom(utcLocal));
        utcZero.setTimeInMillis(utcLocal.getTimeInMillis());
        Log.e(TAG, "utcLocal->GMT  = " + getCalLogStr(utcZero));

        // UTC Local +9
        Calendar utcLocal9 = Calendar.getInstance(TimeZone.getDefault());
        utcLocal9.set(1970, 0, 1, 9, 0, 0);
        utcLocal9.set(Calendar.MILLISECOND, 0);
        Log.e(TAG, "utcLocal9 = " + getCalLogStr(utcLocal9));
        Log.e(TAG, "utcLocal9.getTimeInMillis() = " + utcLocal9.getTimeInMillis());
        Log.e(TAG, "getUtcMillisFrom(utcLocal9) = " + getUtcMillisFrom(utcLocal9));
        utcZero.setTimeInMillis(utcLocal9.getTimeInMillis());
        Log.e(TAG, "utcLocal9->GMT = " + getCalLogStr(utcZero));
        Log.e(TAG, "--------------------------------------------------------");
*/








        Calendar gmtCal = Calendar.getInstance(TimeZone.getTimeZone("GMT"));
        gmtCal.set(Calendar.MILLISECOND, 0);
        Calendar curLocalCal = Calendar.getInstance();
        curLocalCal.set(Calendar.MILLISECOND, 0);

        Log.e(TAG, "### DST IN / Local->UTC");
        int min = 30;
        for (int hour = 0; hour < 5; ++hour) {
            // Current Local.
            curLocalCal.set(1948, 4, 2, hour, min, 0);
            gmtCal.setTimeInMillis(curLocalCal.getTimeInMillis());
            Log.e(TAG, "INPUT= 1948/05/02 " + String.format("%02d", hour) + ":" + String.format("%02d", min) + ":00"
                    + " / "
                    + "LOCAL= " + getCalLogStr(curLocalCal)
                    + " / "
                    + "GMT =" + getCalLogStr(gmtCal));
//            Log.e(TAG, "curLocalCal.getTimeInMillis()    = " + curLocalCal.getTimeInMillis());
//            Log.e(TAG, "curLocalCal.getTime().getTime()  = " + curLocalCal.getTime().getTime());
//            Log.e(TAG, "curLocalCal.ZONE_OFFSET          = " + curLocalCal.get(Calendar.ZONE_OFFSET) / 1000 / 60 / 60);
//            Log.e(TAG, "curLocalCal.DST_OFFSET           = " + curLocalCal.get(Calendar.DST_OFFSET) / 1000 / 60 / 60);
        }

        Log.e(TAG, "### DST IN / UTC->Local");
        for (int i = 0; i < 6; ++i) {
            long diff = 1 * 60 * 60 * 1000;
            long utcMillis = -683802000000L + diff * i - min * 60 * 1000;
            curLocalCal.setTimeInMillis(utcMillis);
            gmtCal.setTimeInMillis(curLocalCal.getTimeInMillis());
            Log.e(TAG, "INPUT= " + utcMillis + " / "
                    + "GMT= " + getCalLogStr(gmtCal)
                    + " / "
                    + "LOCAL= " + getCalLogStr(curLocalCal));
        }

        Log.e(TAG, "### DST OUT / Local->UTC");
        for (int hour = 22; hour < 24 ; ++hour) {
            curLocalCal.set(1948, 8, 10, hour, min, 0);
            gmtCal.setTimeInMillis(curLocalCal.getTimeInMillis());
            Log.e(TAG, "INPUT= 1948/09/10 " + String.format("%02d", hour) + ":" + String.format("%02d", min) + ":00"
                    + " / "
                    + "LOCAL= " + getCalLogStr(curLocalCal)
                    + " / "
                    + "GMT =" + getCalLogStr(gmtCal));
        }
        for (int hour = 0; hour < 5 ; ++hour) {
            curLocalCal.set(1948, 8, 11, hour, min, 0);
            gmtCal.setTimeInMillis(curLocalCal.getTimeInMillis());
            Log.e(TAG, "INPUT= 1948/09/11 " + String.format("%02d", hour) + ":" + String.format("%02d", min) + ":00"
                    + " / "
                    + "LOCAL =" + getCalLogStr(curLocalCal)
                    + " / "
                    + "GMT =" + getCalLogStr(gmtCal));
        }

        Log.e(TAG, "### DST OUT / UTC->Local");
        for (int i = 0; i < 6; ++i) {
            long diff = 1 * 60 * 60 * 1000;
            long utcMillis = -672404400000L + diff * i - min * 60 * 1000;
            curLocalCal.setTimeInMillis(utcMillis);
            gmtCal.setTimeInMillis(curLocalCal.getTimeInMillis());
            Log.e(TAG, "INPUT= " + utcMillis + " / "
                    + "GMT= " + getCalLogStr(gmtCal)
                    + " / "
                    + "LOCAL= " + getCalLogStr(curLocalCal));
        }

        Log.e(TAG, "check() : X");
    }

    private static long getUtcMillisFrom(Calendar cal) {
        long localMillis = cal.getTimeInMillis();
        long tzOffset = cal.get(Calendar.ZONE_OFFSET);
        long dstOffset = cal.get(Calendar.DST_OFFSET);
        return localMillis - tzOffset - dstOffset;
    }

    private static String getCalLogStr(Calendar cal) {
        int year = cal.get(Calendar.YEAR);
        int month = cal.get(Calendar.MONTH);
        int day = cal.get(Calendar.DAY_OF_MONTH);
        int hour = cal.get(Calendar.HOUR_OF_DAY);
        int min = cal.get(Calendar.MINUTE);
        int sec = cal.get(Calendar.SECOND);
        int tz = cal.get(Calendar.ZONE_OFFSET);
        int dst = cal.get(Calendar.DST_OFFSET);

        StringBuilder strBuilder = new StringBuilder()
                .append(year).append('/')
                .append(String.format("%02d", month + 1)).append('/')
                .append(String.format("%02d", day)).append(' ')
                .append(String.format("%02d", hour)).append(':')
                .append(String.format("%02d", min)).append(':')
                .append(String.format("%02d", sec)).append(' ')
                .append((0 <= tz) ? '+' : "").append(tz / 1000 / 60 / 60).append(' ')
                .append("DST+").append(dst / 1000 / 60 / 60);
        return strBuilder.toString();
    }
}
