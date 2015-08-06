package com.fezrestia.gae.util;

import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.http.HttpServletResponse;

public final class ErrorPage {
    public static void create(
            HttpServletResponse response,
            String message,
            String redirectUrl)
                    throws IOException {
        response.setContentType("text/html");
        response.setCharacterEncoding("utf-8");
        PrintWriter pw = response.getWriter();
        pw.println("<html><body>");
        pw.println("<h2>" + message + "</h2>");
        pw.println("<a href=\"" + redirectUrl + "\">Go back</a>");
        pw.println("</body></html>");
    }
}
