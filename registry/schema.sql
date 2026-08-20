PRAGMA foreign_keys = ON;

-- Une exécution du catalogue peut interroger plusieurs recherches Hugging Face.
CREATE TABLE IF NOT EXISTS search_run (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    error_message TEXT
);

-- Un dépôt Hugging Face peut contenir plusieurs checkpoints distincts.
CREATE TABLE IF NOT EXISTS repository (
    repository_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_url TEXT NOT NULL,
    author TEXT,
    license TEXT,
    pipeline_tag TEXT,
    library_name TEXT,
    downloads INTEGER,
    likes INTEGER,
    revision TEXT,
    last_modified TEXT,
    card_text TEXT,
    discovered_at TEXT NOT NULL,
    last_checked_at TEXT NOT NULL
);

-- Une ligne correspond à un modèle/checkpoint réellement comparable à ASC.
CREATE TABLE IF NOT EXISTS candidate (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    repository_id TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    display_name TEXT NOT NULL,
    architecture TEXT,
    weights_available INTEGER NOT NULL CHECK (weights_available IN (0, 1)),
    pretrained_channel_count INTEGER,
    channel_mode TEXT NOT NULL DEFAULT 'inconnu'
        CHECK (channel_mode IN ('fixes', 'variables', 'inconnu')),
    electrode_positions_available INTEGER
        CHECK (electrode_positions_available IN (0, 1)
               OR electrode_positions_available IS NULL),
    sampling_frequency_hz REAL,
    window_duration_seconds REAL,
    number_of_timepoints INTEGER,
    preprocessing TEXT,
    pretraining_dataset TEXT,
    source_task TEXT,
    asc_32_status TEXT NOT NULL DEFAULT 'à vérifier'
        CHECK (asc_32_status IN ('oui', 'non', 'à adapter', 'à vérifier')),
    compatibility_reason TEXT NOT NULL,
    compatibility_score REAL NOT NULL DEFAULT 0,
    confidence TEXT NOT NULL DEFAULT 'faible'
        CHECK (confidence IN ('forte', 'moyenne', 'faible')),
    metadata_status TEXT NOT NULL DEFAULT 'incomplet',
    inspected_at TEXT NOT NULL,
    FOREIGN KEY (repository_id) REFERENCES repository(repository_id)
        ON DELETE CASCADE,
    UNIQUE (repository_id, checkpoint_path)
);

-- Les fichiers de poids sont normalisés dans une table séparée.
CREATE TABLE IF NOT EXISTS weight_file (
    weight_file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes INTEGER,
    download_url TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id)
        ON DELETE CASCADE,
    UNIQUE (candidate_id, file_path)
);

-- L'ordre des électrodes est scientifiquement important pour un poids spatial.
CREATE TABLE IF NOT EXISTS candidate_electrode (
    candidate_electrode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    electrode_order INTEGER NOT NULL,
    electrode_name TEXT NOT NULL,
    x REAL,
    y REAL,
    z REAL,
    source_url TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id)
        ON DELETE CASCADE,
    UNIQUE (candidate_id, electrode_order)
);

-- Chaque valeur importante conserve sa provenance et sa méthode d'extraction.
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    source_url TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    confidence TEXT NOT NULL
        CHECK (confidence IN ('forte', 'moyenne', 'faible')),
    collected_at TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id)
        ON DELETE CASCADE
);

-- Relation plusieurs-à-plusieurs entre une recherche et les candidats trouvés.
CREATE TABLE IF NOT EXISTS search_result (
    run_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    result_rank INTEGER NOT NULL,
    PRIMARY KEY (run_id, candidate_id, query_text),
    FOREIGN KEY (run_id) REFERENCES search_run(run_id) ON DELETE CASCADE,
    FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_candidate_compatibility
    ON candidate(asc_32_status, compatibility_score DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_candidate_field
    ON evidence(candidate_id, field_name);

-- La vue est recréée sans toucher aux données afin que les évolutions de sa
-- présentation soient appliquées aux bases déjà existantes.
DROP VIEW IF EXISTS asc_model_catalog;

-- Cette vue présente exactement les colonnes demandées pour la réunion.
CREATE VIEW asc_model_catalog AS
SELECT
    c.candidate_id AS candidate_id,
    c.display_name AS "Nom du modèle",
    CASE c.weights_available WHEN 1 THEN 'oui' ELSE 'non' END
        AS "Poids réellement disponibles",
    COALESCE(CAST(c.pretrained_channel_count AS TEXT), 'à vérifier')
        AS "Nombre de canaux du pré-entraînement",
    c.channel_mode AS "Canaux fixes ou variables",
    COALESCE(
        (
            SELECT GROUP_CONCAT(
                CASE
                    WHEN ordered_electrodes.x IS NOT NULL
                     AND ordered_electrodes.y IS NOT NULL
                     AND ordered_electrodes.z IS NOT NULL
                    THEN printf(
                        '%s (%.6f; %.6f; %.6f)',
                        ordered_electrodes.electrode_name,
                        ordered_electrodes.x,
                        ordered_electrodes.y,
                        ordered_electrodes.z
                    )
                    ELSE ordered_electrodes.electrode_name
                END,
                ', '
            )
            FROM (
                SELECT electrode_name, x, y, z
                FROM candidate_electrode ce
                WHERE ce.candidate_id = c.candidate_id
                ORDER BY electrode_order
            ) AS ordered_electrodes
        ),
        'à vérifier'
    ) AS "Noms/positions des électrodes utilisés",
    COALESCE(CAST(c.sampling_frequency_hz AS TEXT), 'à vérifier')
        AS "Fréquence de pré-entraînement",
    COALESCE(CAST(c.window_duration_seconds AS TEXT), 'à vérifier')
        AS "Durée des fenêtres",
    COALESCE(c.preprocessing, 'à vérifier') AS "Prétraitement",
    COALESCE(c.pretraining_dataset, 'à vérifier')
        AS "Dataset de pré-entraînement",
    c.asc_32_status AS "Peut recevoir mes 32 canaux ASC",
    c.compatibility_reason AS "Justification",
    c.compatibility_score AS "Score de compatibilité",
    c.confidence AS "Confiance des métadonnées",
    r.model_url AS "URL",
    r.license AS "Licence",
    c.checkpoint_path AS "Chemin du checkpoint"
FROM candidate c
JOIN repository r ON r.repository_id = c.repository_id;
