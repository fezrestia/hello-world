package com.fezrestia.gae.mail;

import java.io.IOException;
import java.net.URL;
import java.util.Properties;
import java.util.logging.Logger;

import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.internet.MimeMessage;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.appengine.api.mail.MailService;
import com.google.appengine.api.mail.MailServiceFactory;

@SuppressWarnings("serial")
public class JavaMailReceiver extends HttpServlet {
    public static final String TAG = JavaMailReceiver.class.getSimpleName();
    private static final Logger LOGGER = Logger.getLogger(TAG);

    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        LOGGER.info("doPost() : E");

        // Reply to address.
        URL url = new URL(request.getRequestURL().toString());
        String sender = "echo@" + url.getHost().replace("appspot",  "appspotmail");
        LOGGER.info("sender = " + sender);

        // Session.
        Properties prop = new Properties();
        Session session = Session.getInstance(prop, null);

        try {
            // Received object.
            MimeMessage receivedMsg = new MimeMessage(session, request.getInputStream());

            // Send message.
            MailService.Message sendMsg = new MailService.Message();
            sendMsg.setSubject(receivedMsg.getSubject());
            sendMsg.setTo(receivedMsg.getFrom()[0].toString());
            sendMsg.setSender(sender);

            // Check MIME.
            String mimeType = receivedMsg.getContentType();
            if (mimeType == null) {
                // NOP. Unexpected.
            } else if (mimeType.startsWith("text/plain")) {
                // Normal text mail.
                sendMsg.setTextBody((String) receivedMsg.getContent());
            } else if (mimeType.startsWith("multipart/alternative")) {

                //TODO:

            } else {
                // Unknown format.
                LOGGER.info("Unknown format : " + receivedMsg.getContentType());
                sendMsg.setTextBody("Unknown mail format : " + receivedMsg.getContentType());
            }

            // Send.
            MailServiceFactory.getMailService().send(sendMsg);
        } catch (MessagingException e) {
            e.printStackTrace();
            throw new IOException(e);
        }

        LOGGER.info("doPost() : X");
    }
}
