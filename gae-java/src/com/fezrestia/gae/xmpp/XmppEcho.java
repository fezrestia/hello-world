package com.fezrestia.gae.xmpp;

import java.io.IOException;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.appengine.api.xmpp.JID;
import com.google.appengine.api.xmpp.Message;
import com.google.appengine.api.xmpp.MessageBuilder;
import com.google.appengine.api.xmpp.MessageType;
import com.google.appengine.api.xmpp.SendResponse;
import com.google.appengine.api.xmpp.XMPPService;
import com.google.appengine.api.xmpp.XMPPServiceFactory;

@SuppressWarnings("serial")
public class XmppEcho extends HttpServlet {
    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {

        // XMPP service.
        XMPPService xmpp = XMPPServiceFactory.getXMPPService();

        // Parse received message.
        Message receivedMsg = xmpp.parseMessage(request);

        // Get reply address.
        String xmppAddress = receivedMsg.getFromJid().getId();

        // Generate JID.
        JID jid = new JID(xmppAddress);

        // Create message.
        Message sendMsg = new MessageBuilder()
                .withMessageType(MessageType.CHAT)
                .withRecipientJids(jid)
                .withBody(receivedMsg.getBody())
                .build();

        // Send.
        SendResponse xmppResponse = xmpp.sendMessage(sendMsg);

        // Check.
        SendResponse.Status sendStatus = xmppResponse.getStatusMap().get(jid);
        if (sendStatus != SendResponse.Status.SUCCESS) {
            throw new IOException("Failed to send message : " + sendStatus);
        }
    }
}
