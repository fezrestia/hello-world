create table posts (
    id serial primary key,
    user_id int not null,
    title varchar(255) not null,
    body text check(length(body) > 5),
    is_draft boolean default TRUE,
    created timestamp default 'now'
);

insert into posts (user_id, title, body) values
    (1, 'Title 1', 'Body 1'),
    (1, 'Title 2', 'Body 2'),
    (2, 'Title 3', 'Body 3'),
    (5, 'Title 4', 'Body 4'),
    (4, 'Title 5', 'Body 5')

-- Comment
/*
    Comments

Num : inteer(int), real, serial
Char: char(5), varchar(255), text
bool: boolean TRUE or FALSE, t or f
Date: date, time, timestamp

Constraint
    not null
    unique
    check
    default
    primary key (not null, unique)

*/
