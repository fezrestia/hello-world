drop table if exists posts;

create table posts (
    id integer primary key,
    title text,
    body text,
    created_at,
    updated_at
);

insert into posts(id, title, body) values (1, "title1", "hello1");
insert into posts(id, title, body) values (2, "title2", "hello2");
insert into posts(id, title, body) values (3, "title3", "hello3");

drop table if exists comments;

create table comments (
    id integer primary key,
    post_id integer,
    body text,
    created_at,
    updated_at
);

insert into comments(id, post_id, body) values (1, 1, "comment1 for 1");
insert into comments(id, post_id, body) values (2, 1, "comment2 for 1");
insert into comments(id, post_id, body) values (3, 2, "comment3 for 2");
insert into comments(id, post_id, body) values (4, 2, "comment4 for 2");
insert into comments(id, post_id, body) values (5, 3, "comment5 for 3");

