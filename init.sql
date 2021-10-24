DROP TABLE IF EXISTS cuts;
DROP TABLE IF EXISTS cut_ids;

CREATE TABLE cut_ids (
    cut_id guid NOT NULL,
    PRIMARY KEY(cut_id)
);

CREATE TABLE cuts (
    cut_id guid NOT NULL,
    cut text NOT NULL,
    PRIMARY KEY(cut_id)
    FOREIGN KEY(cut_id) REFERENCES cut_ids(cut_id)
);

PRAGMA foreign_keys = ON;