CREATE TABLE IF NOT EXISTS user_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    filename_hash TEXT NOT NULL,
    sheet_name TEXT,
    original_request TEXT,
    generated_command_json TEXT,
    parser_confidence REAL,
    parser_source TEXT,
    action_type TEXT,
    success INTEGER NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER,
    original_request TEXT,
    incorrect_command_json TEXT,
    corrected_command_json TEXT,
    correction_type TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES user_requests(id)
);

CREATE TABLE IF NOT EXISTS learned_synonyms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL,
    mapped_concept TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(phrase, mapped_concept)
);

CREATE TABLE IF NOT EXISTS learned_column_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_column_name TEXT NOT NULL,
    standard_concept TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(raw_column_name, standard_concept)
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    username TEXT,
    user_role TEXT NOT NULL,
    action_type TEXT,
    columns_affected TEXT,
    row_count_affected INTEGER,
    success INTEGER NOT NULL,
    source TEXT
);
