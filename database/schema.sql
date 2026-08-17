PRAGMA foreign_keys = ON;

-- Les signaux EEG restent dans les fichiers .npy. Cette table conserve
-- uniquement les métadonnées nécessaires pour les retrouver et les filtrer.
CREATE TABLE IF NOT EXISTS epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dyad_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    condition TEXT NOT NULL CHECK (condition IN ('YO', 'YF')),
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    number_of_participants INTEGER NOT NULL DEFAULT 2,
    number_of_channels INTEGER NOT NULL,
    number_of_timepoints INTEGER NOT NULL,
    sampling_frequency_hz REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    matrix_dtype TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dyad_id, filename)
);

-- Une ligne décrit le résultat d'un diagnostic pour un participant et une
-- électrode. diagnostic_version permet de comparer plusieurs méthodologies.
CREATE TABLE IF NOT EXISTS diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch_id INTEGER NOT NULL,
    participant TEXT NOT NULL CHECK (participant IN ('P1', 'P2')),
    electrode TEXT NOT NULL,
    max_absolute REAL,
    peak_to_peak REAL,
    n_above_threshold INTEGER NOT NULL DEFAULT 0,
    absolute_threshold REAL,
    relative_score REAL,
    is_outlier INTEGER NOT NULL DEFAULT 0 CHECK (is_outlier IN (0, 1)),
    diagnostic_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (epoch_id) REFERENCES epochs(id) ON DELETE CASCADE,
    UNIQUE (epoch_id, participant, electrode, diagnostic_version)
);

-- Cette table versionne les checkpoints et le pipeline associé. Une
-- prédiction doit toujours pouvoir être reliée au modèle qui l'a produite.
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    architecture TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    validation_protocol TEXT NOT NULL DEFAULT 'Leave-One-Dyad-Out',
    mean_validation_accuracy REAL,
    is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (name, version)
);

-- L'historique d'inférence est stocké sans dupliquer les signaux EEG.
-- epoch_id est nullable pour autoriser plus tard la prédiction d'un fichier
-- déposé par un utilisateur et absent du dataset de recherche.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch_id INTEGER,
    source_filename TEXT NOT NULL,
    participant TEXT NOT NULL CHECK (participant IN ('P1', 'P2')),
    model_id INTEGER NOT NULL,
    predicted_class TEXT NOT NULL CHECK (predicted_class IN ('YO', 'YF')),
    probability_yo REAL NOT NULL CHECK (probability_yo BETWEEN 0 AND 1),
    probability_yf REAL NOT NULL CHECK (probability_yf BETWEEN 0 AND 1),
    true_class TEXT CHECK (true_class IN ('YO', 'YF') OR true_class IS NULL),
    inference_time_ms REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (epoch_id) REFERENCES epochs(id) ON DELETE SET NULL,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE RESTRICT,
    CHECK (ABS((probability_yo + probability_yf) - 1.0) < 0.0001)
);

CREATE INDEX IF NOT EXISTS idx_epochs_dyad
    ON epochs(dyad_id);

CREATE INDEX IF NOT EXISTS idx_epochs_condition
    ON epochs(condition);

CREATE INDEX IF NOT EXISTS idx_diagnostics_outlier
    ON diagnostics(is_outlier);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions(created_at);
