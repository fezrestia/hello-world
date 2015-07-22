package com.fezrestia.gae.memo;

import java.io.IOException;
import java.text.MessageFormat;
import java.util.List;
import java.util.TimeZone;

import javax.jdo.PersistenceManager;
import javax.jdo.Query;
import javax.servlet.http.*;

import com.fezrestia.gae.util.PMF;

@SuppressWarnings("serial")
public class MainPage extends HttpServlet {

    private static final String HEAD =
            "<html>\n" +
            "<body>\n" +
            "  <span style=\"font-size: 200%\">MEMO</span>\n" +
            "    <form action=\"/memoNew\" method=\"post\">\n" +
            "      <div>\n" +
            "        <textarea name=\"content\" rows=\"2\" cols=\"40\"></textarea>" +
            "      </div>\n" +
            "      <input type=\"submit\" value=\"SUBMIT\" />\n" +
            "    </form>\n";

    private static final String MEMO_TEMPLATE =
            "    <div>\n" +
            "      <span class=\"date\"> {0, time} {0, date} </span>\n" +
            "      <pre>{1}</pre>\n" +
            "    </div>\n";

    private static final String TAIL =
            "</body>\n" +
            "</html>\n";

    static {
        TimeZone.setDefault(TimeZone.getTimeZone("JST"));
    }

    private String render(List<Memo> memoList) {
        StringBuilder builder = new StringBuilder().append(HEAD);
        for (Memo eachMemo : memoList) {
            builder.append(MessageFormat.format(
                    MEMO_TEMPLATE,
                    eachMemo.getDate(),
                    eachMemo.getContent()));
        }
        builder.append(TAIL);

        return builder.toString();
    }


    @Override
    public void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        PersistenceManager pm = null;
        try {
            pm = PMF.get().getPersistenceManager();
            Query query = pm.newQuery(Memo.class);
            query.setOrdering("date desc");

            List<Memo> memos = (List<Memo>) query.execute();

            resp.setContentType("text/html");
            resp.setCharacterEncoding("utf-8");
            resp.getWriter().print(render(memos));
        } finally {
            if (pm != null && !pm.isClosed()) {
                pm.close();
            }
        }
    }
}
