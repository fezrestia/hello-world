package com.fezrestia.gae.parsehtml;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.util.List;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import net.htmlparser.jericho.CharacterReference;
import net.htmlparser.jericho.Element;
import net.htmlparser.jericho.HTMLElementName;
import net.htmlparser.jericho.Source;

@SuppressWarnings("serial")
public final class HtmlParser extends HttpServlet {

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
            throws IOException {
        response.setContentType("text/html; charset=utf8");
        PrintWriter writer = response.getWriter();
        writer.println("<table width=\"800\" style=\"table-layout: fixed; "
                + "word-wrap: break-word;\">");
        writer.println("<tr><th>link</th><th>label</ht></hr>");

        // Get URL from parameters.
        String url = request.getParameter("url");

        // Set source.
        Source source = new Source(new URL(url));

        // Sdearch <a href="xxx">yyy</a>.
        List<Element> elementList = source.getAllElements(HTMLElementName.A);
        for (Element eachElement : elementList) {
            writer.println("<tr><td>");
            writer.println(eachElement.getAttributeValue("href"));
            writer.println("</td><td>");
            writer.println(escape(CharacterReference.decodeCollapseWhiteSpace(eachElement.getContent())));
            writer.println("</td></tr>");
        }
        writer.println("</table>");
    }

    private static String escape(String input) {
        String ret = new String(input);
        ret = ret.replace("\"",  "&quot;");
        ret = ret.replace("&",  "&amp;");
        ret = ret.replace(">", "&gt;");
        ret = ret.replace("<", "&lt;");
        return ret;
    }
}
