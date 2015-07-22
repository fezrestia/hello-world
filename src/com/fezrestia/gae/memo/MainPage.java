package com.fezrestia.gae.memo;

import java.io.IOException;
import java.util.List;
import java.util.TimeZone;

import javax.jdo.PersistenceManager;
import javax.jdo.Query;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.context.Context;

import com.fezrestia.gae.util.PMF;
import com.fezrestia.gae.velocity.Renderer;

@SuppressWarnings("serial")
public class MainPage extends HttpServlet {
    static {
        TimeZone.setDefault(TimeZone.getTimeZone("JST"));
    }

    @Override
    public void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        PersistenceManager pm = PMF.get().getPersistenceManager();

        Query query = pm.newQuery(Memo.class);
        query.setOrdering("date desc");

        List<Memo> memos = (List<Memo>) query.execute();

        Context context = new VelocityContext();
        context.put("memos",  memos);

        resp.setContentType("text/html");
        resp.setCharacterEncoding("utf-8");

        Renderer.render("WEB-INF/memoMainPage.vm", context, resp.getWriter());
    }
}
