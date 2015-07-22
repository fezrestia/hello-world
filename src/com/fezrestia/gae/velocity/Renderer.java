package com.fezrestia.gae.velocity;

import java.io.IOException;
import java.io.Writer;
import java.text.DateFormat;
import java.util.TimeZone;
import java.util.logging.Logger;

import org.apache.velocity.Template;
import org.apache.velocity.app.Velocity;
import org.apache.velocity.context.Context;
import org.apache.velocity.runtime.log.JdkLogChute;

public class Renderer {
    private static final Logger LOGGER = Logger.getAnonymousLogger();
    private static boolean mIsInitialized = false;

    private static final DateFormat DATE_TIME_FORMAT = DateFormat.getDateTimeInstance();

    static {
        DATE_TIME_FORMAT.setTimeZone(TimeZone.getTimeZone("JST"));
    }

    private static void initializeVelocity() throws Exception {
        Velocity.setProperty(Velocity.RUNTIME_LOG_LOGSYSTEM, new JdkLogChute());
        Velocity.init();
        mIsInitialized = true;
    }

    public static void render(String filename, Context context, Writer writer)
            throws IOException {
        try {
            synchronized (LOGGER) {
                if (!mIsInitialized) {
                    initializeVelocity();
                }

                context.put("_datetimeFormat", DATE_TIME_FORMAT);

                Template template = Velocity.getTemplate(filename);
                template.merge(context, writer);
            }
        } catch (Exception e) {
            throw new IOException(e);
        }
    }
}
