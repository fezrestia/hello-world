package com.fezrestia.gae.imagemail;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Properties;

import javax.mail.MessagingException;
import javax.mail.Multipart;
import javax.mail.Part;
import javax.mail.Session;
import javax.mail.internet.MimeMessage;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.fezrestia.gae.util.ImageUtil;
import com.google.appengine.api.mail.MailService;
import com.google.appengine.api.mail.MailServiceFactory;

@SuppressWarnings("serial")
public class ImageMailHandler extends HttpServlet {

    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {

        // Sender address.
        URL url = new URL(request.getRequestURL().toString());
        String senderAddress = "echo@" + url.getHost().replace("appspot", "appspotmail");

        // Image display servlet.
        String imageDisplayUrl = url.getProtocol() + "://" + url.getHost() + "/utilImageDisplay";

        Properties properties = System.getProperties();
        Session session = Session.getInstance(properties, null);

        // Received message.
        MimeMessage receivedMsg;
        try {
            receivedMsg = new MimeMessage(session, request.getInputStream());
        } catch (MessagingException e) {
            e.printStackTrace();
            return;
        }

        // Response message head.
        StringBuilder sb = new StringBuilder("<html><body><h3>Picture uploaded</h3><ul>");

        try {
            // Parse received message and create response message.
            createMessage(receivedMsg, sb, imageDisplayUrl);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        } catch (MessagingException e) {
            e.printStackTrace();
            return;
        }

        // Response message tail.
        sb.append("</ul></body></html>");

        // Generate response message.
        MailService.Message sendMsg;
        try {
            sendMsg = new MailService.Message();
            sendMsg.setSubject(receivedMsg.getSubject());
            sendMsg.setTo(receivedMsg.getFrom()[0].toString());
            sendMsg.setHtmlBody(sb.toString());
            sendMsg.setSender(senderAddress);
        } catch (MessagingException e) {
            e.printStackTrace();
            return;
        }

        // Do send.
        MailServiceFactory.getMailService().send(sendMsg);
    }

    private void createMessage(
            Part receivedMsg,
            StringBuilder responseMsgSb,
            String imageDisplayUrl)
                    throws IOException, MessagingException {
        if (receivedMsg == null || receivedMsg.getContentType() == null) {
            // NOP. Unexpected.
            throw new IllegalArgumentException("ReceivedMsg or ContentType is NULL.");
        } else if (receivedMsg.getContentType().startsWith("text/")) {
            // Ignore.
            return;
        } else if (receivedMsg.getContentType().startsWith("image/")) {
            // Image file.
            // Expected : org.apache.geronimo.mail.util.Base64DecoderStream
            Object content = receivedMsg.getContent();
            long id = ImageUtil.storeImageToDataStore(ImageUtil.readBytes((InputStream) content));

            // Create link.
            responseMsgSb.append("<li><a href=\"" + imageDisplayUrl)
                    .append("?imageId=" + id + "\">" + "image</a></li>");
            return;
        } else if (receivedMsg.getContentType().startsWith("multipart/")) {
            // Read recursively.
            Multipart mp = (Multipart) receivedMsg.getContent();
            for (int i = 0;i < mp.getCount(); ++i) {
                createMessage(mp.getBodyPart(i), responseMsgSb, imageDisplayUrl);
            }
            return;
        } else {
            // Unexpected content type.
            throw new IOException("Unexpected content type.");
        }
    }
}
