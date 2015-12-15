package com.fezrestia.gae.mail;

import java.io.IOException;
import java.util.Properties;

import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.AddressException;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;
import javax.mail.internet.MimeUtility;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.appengine.api.users.User;
import com.google.appengine.api.users.UserService;
import com.google.appengine.api.users.UserServiceFactory;

@SuppressWarnings("serial")
public class JavaMailSender extends HttpServlet {

    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        // User.
        UserService userService = UserServiceFactory.getUserService();
        User user = userService.getCurrentUser();

        // Log-IN check.
        if (user == null) {
            // Not log-IN.
            response.setContentType("text/plain");
            response.getWriter().println("Not Log-IN.");
            return;
        }

        // Mail session.
        Properties props = new Properties();
        Session session = Session.getDefaultInstance(props, null);

        // Contents.
        String msgBody = "Hello from GAE/J";
        String msgSubject = MimeUtility.encodeText("Hello ??", "ISO-2022-JP", "B");

        try {
            Message msg = new MimeMessage(session);

            // Header.
            msg.setFrom(new InternetAddress(user.getEmail(), user.getNickname()));
            msg.addRecipient(
                    Message.RecipientType.TO,
                    new InternetAddress(user.getEmail(), user.getNickname()));
            msg.setSubject(msgSubject);

            // Body.
            MimeBodyPart mbText = new MimeBodyPart();
            mbText.setText(msgBody);

            // Attachment.
            MimeBodyPart mbAttachment = new MimeBodyPart();
            mbAttachment.attachFile("WEB-INF/kamo.jpg");

            // Bind.
            MimeMultipart mm = new MimeMultipart();
            mm.addBodyPart(mbText);
            mm.addBodyPart(mbAttachment);
            msg.setContent(mm);

            // Send.
            Transport.send(msg);
        } catch (AddressException e) {
            e.printStackTrace();
            throw new IOException(e);
        } catch (MessagingException e) {
            e.printStackTrace();
            throw new IOException(e);
        }

        // Ret.
        response.setContentType("text/plain");
        response.getWriter().println("Send mail DONE.");
    }
}
