create table users (
    id serial primary key,
    name varchar(255),
    score real,
    team varchar(255)
);

insert into users (name, score, team) values
    ('taguchi', 5.5, 'red'),
    ('fkoji', 8.3, 'blue'),
    ('dotinstall', 2.2, 'blue'),
    ('sasaki', 5.0, 'green'),
    ('sakaki', 4.6, 'red'),
    ('kimura', 4.7, 'green');

