package com.fezrestia.gae.mail;

import java.io.IOException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.fezrestia.util.ByteUtil;
import com.google.appengine.api.mail.MailService;
import com.google.appengine.api.mail.MailService.Attachment;
import com.google.appengine.api.mail.MailService.Message;
import com.google.appengine.api.mail.MailServiceFactory;
import com.google.appengine.api.users.User;
import com.google.appengine.api.users.UserService;
import com.google.appengine.api.users.UserServiceFactory;

@SuppressWarnings("serial")
public class LowLevelMailSender extends HttpServlet {

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

        // Message.
        Message message = new Message();
        message.setSender(user.getEmail());
        message.setTo(user.getEmail());
//        message.setReplyTo(user.getEmail());
//        message.setBcc(user.getEmail());
        message.setSubject("Hello ?");
        message.setTextBody("Hello from GAE.");

        // Attachment.
        Attachment picture = new Attachment(
                "KAMO.JPG",
                ByteUtil.getBytesFrom("WEB-INF/kamo.jpg"));
        message.setAttachments(picture);

        // Service.
        MailService ms = MailServiceFactory.getMailService();
        // Send.
        ms.send(message);

        // Ret.
        response.setContentType("text/plain");
        response.getWriter().println("Send mail DONE.");
    }
}
